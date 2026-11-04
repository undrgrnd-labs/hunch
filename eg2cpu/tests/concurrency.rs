//! Concurrency and early exit: scheduled results equal direct passes, and resuming an early exit equals a full pass.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn concurrent_callers_get_their_own_results() {
    let Some(model) = common::model_or_skip("concurrent_callers_get_their_own_results") else { return };
    let base =
        model.tokenize(&"The agent called read_file, got an error, and retried the same call. ".repeat(80)).unwrap();
    // Distinct texts of varied length (offset and length differ per request).
    let text = |i: usize| &base[i % 13..i % 13 + [7, 30, 90, 200][i % 4]];
    let want: Vec<Vec<f32>> = (0..56).map(|i| model.forward_batch_with(&[text(i)], 1).remove(0).embedding).collect();

    std::thread::scope(|s| {
        // 16 callers sending single requests, while another sends a 40-text batch.
        let singles: Vec<_> = (0..16)
            .map(|c| {
                let (m, want) = (&model, &want);
                s.spawn(move || {
                    for r in 0..2 {
                        let i = c + 16 * r; // a different text each round
                        assert_eq!(m.forward(text(i)).embedding, want[i], "single request {i}");
                    }
                })
            })
            .collect();
        let batch = s.spawn(|| {
            let ids: Vec<&[u32]> = (16..56).map(text).collect();
            for (k, out) in model.forward_batch(&ids).into_iter().enumerate() {
                assert_eq!(out.embedding, want[16 + k], "batch item {k}");
            }
        });
        singles.into_iter().for_each(|h| h.join().unwrap());
        batch.join().unwrap();
    });

    // A bad request panics in its own caller and leaves the scheduler usable.
    let bad = catch_unwind(AssertUnwindSafe(|| model.forward(&[u32::MAX])));
    assert!(bad.is_err());
    assert_eq!(model.forward(text(3)).embedding, want[3]);
}

#[test]
fn early_exit_then_resume_matches_a_full_pass() {
    let Some(model) = common::model_or_skip("early_exit_then_resume_matches_a_full_pass") else { return };
    let texts = ["bash: git status\nnothing to commit", "read: src/main.rs\nfn main() { println!(\"hi\"); }"];
    let full = model.embed_batch(&texts).unwrap();
    let parts = model.embed_batch_partial(&texts, 8).unwrap();
    for (p, f) in parts.iter().zip(&full) {
        assert_eq!(p.layers_done(), 9);
        assert_eq!(p.layer_means.len(), 9);
        assert_eq!(p.layer_means[8], f.layer_means[8], "layer 8 means of an early exit equal the full pass's");
    }
    let resumed = model.resume_batch(parts);
    for (r, f) in resumed.iter().zip(&full) {
        assert_eq!(r.embedding, f.embedding);
        assert_eq!(r.layer_means, f.layer_means);
    }
}
