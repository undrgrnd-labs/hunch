//! Compare the image path against PyTorch fp32 CPU outputs from scripts/make_image_reference.py: the resized
//! pixels, the soft tokens from the vision tower, all 24 layer means, and the final embedding, plus the
//! text-image similarities of a few captions.

mod common;

use common::{dot, floats, rel_err};
use serde_json::Value;

#[test]
fn matches_pytorch_image_reference() {
    let Some(model) = common::model_or_skip("matches_pytorch_image_reference") else { return };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}image_reference.json")).unwrap()).unwrap();
    let captions: Vec<&str> = fixture["captions"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let text_embs: Vec<Vec<f32>> = captions.iter().map(|c| model.embed(c).unwrap().embedding).collect();
    let d = model.cfg.hidden;
    let mut lowest = 1.0f32;
    for (case, want_sims) in fixture["cases"].as_array().unwrap().iter().zip(fixture["similarity"].as_array().unwrap())
    {
        let file = case["file"].as_str().unwrap();
        let budget = case["max_soft_tokens"].as_u64().unwrap() as usize;
        let bytes = std::fs::read(format!("{dir}{file}")).unwrap();
        // JPEG decoders may differ by a level here and there (libjpeg-turbo in PIL, jpeg-decoder here); PNG is exact.
        let lossless = file.ends_with(".png");

        let img = model.preprocess_image(&bytes, budget).unwrap();
        let size: Vec<usize> = case["size"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        assert_eq!([img.height, img.width], [size[0], size[1]], "{file}: resized size");
        let sum: u64 = img.data.iter().map(|&b| b as u64).sum();
        let idx = case["pixel_index"].as_array().unwrap();
        let px_diff = idx
            .iter()
            .zip(case["pixel_values"].as_array().unwrap())
            .map(|(i, v)| (img.data[i.as_u64().unwrap() as usize] as i64 - v.as_i64().unwrap()).abs())
            .max()
            .unwrap();
        let want_sum = case["pixel_sum"].as_u64().unwrap();
        if lossless {
            assert_eq!(sum, want_sum, "{file}: pixel sum");
            assert_eq!(px_diff, 0, "{file}: sampled pixels");
        }

        let soft = model.image_soft_tokens(&img).unwrap();
        let n = soft.len() / d;
        let want_ids = case["ids"].as_array().unwrap();
        assert_eq!(n + 4, want_ids.len(), "{file}: soft token count");
        let mean: Vec<f32> = (0..d).map(|j| soft.iter().skip(j).step_by(d).sum::<f32>() / n as f32).collect();
        let head: Vec<f32> = case["soft_tokens_head"].as_array().unwrap().iter().flat_map(floats).collect();
        let soft_err = rel_err(&soft[..2 * d], &head).max(rel_err(&mean, &floats(&case["soft_tokens_mean"])));

        let out = model.embed_soft_tokens(&soft);
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
            "{file:<26} {budget:>4} {:>4}x{:<4} {n:>3} soft  px sum {:+} max|d| {px_diff}  soft rel {soft_err:.1e}  \
             layers rel {worst_layer:.1e}  cos {cos:.7}  sims |d| {sim_err:.1e}",
            img.height,
            img.width,
            sum as i64 - want_sum as i64
        );
        // Lossless images match to fp32 rounding. bf16 matmuls (EG2_BF16) round the activations through both towers (a
        // synthetic gradient can fall to cosine 0.9998), so they get a looser bar. JPEG decoders differ by a few pixel
        // levels, and by how many depends on the platform (jpeg-decoder's x86-64 and Arm paths differ from each other as
        // well as from libjpeg-turbo), so the drift bar is looser still there; the cosine bar is the same.
        let exact = lossless && eg2cpu::bf16_mode() == eg2cpu::Bf16::Off;
        let (min_cos, max_drift) = match (exact, lossless) {
            (true, _) => (0.99999, 1e-3),
            (false, true) => (0.9995, 2e-2),
            (false, false) => (0.9995, 5e-2),
        };
        assert!(cos > min_cos, "{file}: embedding cosine {cos}");
        assert!(soft_err < max_drift && worst_layer < max_drift, "{file}: drift {soft_err} {worst_layer}");
        let argmax = |x: &[f32]| (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap();
        assert_eq!(argmax(&sims), argmax(&want_sims), "{file}: best caption");
        if lossless {
            lowest = lowest.min(cos);
        }
    }
    println!("lowest cos (lossless images) {lowest:.7}");
}

/// Images and text share the embedding space: each photo is closest to its own caption.
#[test]
fn photos_match_their_captions() {
    let Some(model) = common::model_or_skip("photos_match_their_captions") else { return };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/images/");
    let photos = ["chelsea.png", "coffee.png", "astronaut.png", "rocket.jpg"];
    let captions =
        ["a photo of a cat", "a cup of coffee", "an astronaut in a space suit", "a rocket on the launch pad"];
    let texts: Vec<Vec<f32>> = captions.iter().map(|c| model.embed(c).unwrap().embedding).collect();
    for (i, photo) in photos.iter().enumerate() {
        // the smallest budget is enough to tell them apart
        let img = model.embed_image_with(&std::fs::read(format!("{dir}{photo}")).unwrap(), 70).unwrap();
        let sims: Vec<f32> = texts.iter().map(|t| dot(&img.embedding, t)).collect();
        println!("{photo:<14} {}", sims.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>().join(" "));
        let best = (0..sims.len()).max_by(|&a, &b| sims[a].total_cmp(&sims[b])).unwrap();
        assert_eq!(best, i, "{photo} matched {:?}", captions[best]);
    }
}
