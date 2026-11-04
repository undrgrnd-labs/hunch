//! Compare the audio path against PyTorch fp32 CPU outputs from scripts/make_audio_reference.py: the log-mel
//! features, the soft tokens from the audio tower, all 24 layer means, and the final embedding, plus the
//! speech-text similarities of the utterances' transcripts.

mod common;

use common::{dot, floats, rel_err};
use serde_json::Value;

#[test]
fn matches_pytorch_audio_reference() {
    let Some(model) = common::model_or_skip("matches_pytorch_audio_reference") else {
        return;
    };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}audio_reference.json")).unwrap()).unwrap();
    let captions: Vec<&str> = fixture["captions"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let text_embs: Vec<Vec<f32>> = captions.iter().map(|c| model.embed(c).unwrap().embedding).collect();
    let d = model.cfg.hidden;
    let mut lowest = 1.0f32;
    for (case, want_sims) in fixture["cases"].as_array().unwrap().iter().zip(fixture["similarity"].as_array().unwrap())
    {
        let file = case["file"].as_str().unwrap();
        let repeat = case["repeat"].as_u64().unwrap() as usize;
        let samples = eg2cpu::decode_wav(&std::fs::read(format!("{dir}{file}")).unwrap()).unwrap().repeat(repeat);

        let mel = eg2cpu::log_mel(&samples);
        let frames = mel.len() / 128;
        assert_eq!(frames, case["frames"].as_u64().unwrap() as usize, "{file}: mel frames");
        let idx = case["feature_index"].as_array().unwrap();
        let mel_diff = idx
            .iter()
            .zip(case["feature_values"].as_array().unwrap())
            .map(|(i, v)| (mel[i.as_u64().unwrap() as usize] - v.as_f64().unwrap() as f32).abs())
            .fold(0.0f32, f32::max);
        let mel_mean = mel.iter().map(|&v| v as f64).sum::<f64>() / mel.len() as f64;
        let mean_diff = (mel_mean - case["feature_mean"].as_f64().unwrap()).abs();
        assert!(mel_diff < 1e-4 && mean_diff < 1e-5, "{file}: features differ by {mel_diff} (mean {mean_diff})");

        let soft = model.audio_soft_tokens(&mel).unwrap();
        let n = soft.len() / d;
        let want_ids = case["ids"].as_array().unwrap();
        assert_eq!(n + 4, want_ids.len(), "{file}: soft token count");
        assert_eq!(n, eg2cpu::soft_tokens_for(samples.len()), "{file}: soft_tokens_for");
        let mean: Vec<f32> = (0..d).map(|j| soft.iter().skip(j).step_by(d).sum::<f32>() / n as f32).collect();
        let head: Vec<f32> = case["soft_tokens_head"].as_array().unwrap().iter().flat_map(floats).collect();
        let soft_err = rel_err(&soft[..2 * d], &head).max(rel_err(&mean, &floats(&case["soft_tokens_mean"])));

        let out = model.embed_audio_soft_tokens(&soft);
        let want = floats(&case["embedding"]);
        let cos = dot(&out.embedding, &want);
        let worst_layer = case["layer_means"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&out.layer_means)
            .map(|(w, got)| rel_err(got, &floats(w)))
            .fold(0.0f32, f32::max);
        let sims: Vec<f32> = text_embs.iter().map(|t| dot(&out.embedding, t)).collect();
        let want_sims = floats(want_sims);
        let sim_err = sims.iter().zip(&want_sims).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!(
            "{file:<26} x{repeat} {frames:>5} frames {n:>4} soft  mel |d| {mel_diff:.1e}  soft rel {soft_err:.1e}  \
             layers rel {worst_layer:.1e}  cos {cos:.7}  sims |d| {sim_err:.1e}"
        );
        // bf16 matmuls (EG2_BF16) round the activations through both towers, so they get a looser bar.
        let exact = eg2cpu::bf16_mode() == eg2cpu::Bf16::Off;
        let (min_cos, max_drift) = if exact { (0.99999, 1e-3) } else { (0.9995, 2e-2) };
        assert!(cos > min_cos, "{file}: embedding cosine {cos}");
        assert!(soft_err < max_drift && worst_layer < max_drift, "{file}: drift {soft_err} {worst_layer}");
        let argmax = |x: &[f32]| (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap();
        assert_eq!(argmax(&sims), argmax(&want_sims), "{file}: best transcript");
        lowest = lowest.min(cos);
    }
    println!("lowest cos {lowest:.7}");
}

/// Speech and text share the embedding space: each utterance is closest to its own transcript.
#[test]
fn utterances_match_their_transcripts() {
    let Some(model) = common::model_or_skip("utterances_match_their_transcripts") else {
        return;
    };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}audio_reference.json")).unwrap()).unwrap();
    let transcripts: Vec<&str> = fixture["captions"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let texts: Vec<Vec<f32>> = transcripts.iter().map(|c| model.embed(c).unwrap().embedding).collect();
    for (i, clip) in ["speech_a.wav", "speech_b.wav", "speech_c.wav"].iter().enumerate() {
        let out = model.embed_audio(&std::fs::read(format!("{dir}audio/{clip}")).unwrap()).unwrap();
        let sims: Vec<f32> = texts.iter().map(|t| dot(&out.embedding, t)).collect();
        println!("{clip:<14} {}", sims.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>().join(" "));
        let best = (0..sims.len()).max_by(|&a, &b| sims[a].total_cmp(&sims[b])).unwrap();
        assert_eq!(best, i, "{clip} matched {:?}", transcripts[best]);
    }
}

/// The WAV decoder refuses what it can't read faithfully instead of guessing.
#[test]
fn wav_decoder_refuses_other_rates_and_encodings() {
    let wav = |rate: u32, tag: u16, bits: u16| {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend(16u32.to_le_bytes());
        b.extend(tag.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend(rate.to_le_bytes());
        b.extend((rate * bits as u32 / 8).to_le_bytes());
        b.extend((bits / 8).to_le_bytes());
        b.extend(bits.to_le_bytes());
        b.extend(b"data");
        b.extend(4u32.to_le_bytes());
        b.extend([0, 0x40, 0, 0xc0]);
        b
    };
    assert_eq!(eg2cpu::decode_wav(&wav(16_000, 1, 16)).unwrap(), vec![0.5, -0.5]);
    assert!(eg2cpu::decode_wav(&wav(44_100, 1, 16)).unwrap_err().to_string().contains("resample"));
    assert!(eg2cpu::decode_wav(&wav(16_000, 2, 16)).is_err()); // ADPCM
    assert!(eg2cpu::decode_wav(b"OggS not a wav").is_err());
}
