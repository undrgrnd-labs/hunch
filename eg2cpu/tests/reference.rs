//! Compare against PyTorch fp32 CPU outputs from scripts/make_reference.py.

mod common;

use common::{floats, rel_err};
use serde_json::Value;

#[test]
fn matches_pytorch_reference() {
    let Some(model) = common::model_or_skip("matches_pytorch_reference") else { return };
    let cases: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/reference.json")).unwrap(),
    )
    .unwrap();
    let (mut worst, mut lowest) = (0.0f32, 1.0f32);
    for case in &cases {
        let text = case["text"].as_str().unwrap();
        let want_ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let ids = model.tokenize(text).unwrap();
        assert_eq!(ids, want_ids, "token ids differ for {:?}", &text[..text.len().min(40)]);

        let out = model.forward(&ids);
        let want = floats(&case["embedding"]);
        let cos: f32 = out.embedding.iter().zip(&want).map(|(a, b)| a * b).sum();
        let worst_layer = case["layer_means"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&out.layer_means)
            .map(|(w, got)| rel_err(got, &floats(w)))
            .fold(0.0f32, f32::max);
        println!(
            "{:>5} tokens  cos {:.7}  emb max|d| {:.2e}  worst layer-mean rel err {:.2e}",
            ids.len(),
            cos,
            rel_err(&out.embedding, &want),
            worst_layer
        );
        worst = worst.max(worst_layer);
        lowest = lowest.min(cos);
    }
    // fp32 matches PyTorch to rounding; bf16 matmuls (EG2_BF16=amx|dot|bfmmla) round the activations too, so they get
    // their own, looser bar.
    let (min_cos, max_drift) = if eg2cpu::bf16_mode() == eg2cpu::Bf16::Off { (0.99999, 1e-3) } else { (0.9999, 2e-2) };
    println!(
        "{:?}: lowest cos {lowest:.7} (bar {min_cos}), worst layer drift {worst:.2e} (bar {max_drift:.0e})",
        eg2cpu::bf16_mode()
    );
    assert!(lowest > min_cos, "embedding cosine {lowest}");
    assert!(worst < max_drift, "layer means drift {worst}");
}
