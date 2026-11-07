//! eg2cpu's tokenizer against the Hugging Face `tokenizers` crate on the same `tokenizer.json`: the repository's
//! own text and fixtures, long inputs past the truncation point, and random mixes of scripts, emoji, whitespace,
//! control characters, rare code points, and whole or broken special tokens.

use std::path::Path;
use std::time::{Duration, Instant};

use eg2cpu::{MAX_TOKENS, Tokenizer};
use serde_json::Value;

fn tokenizer_json() -> Option<std::path::PathBuf> {
    let path = eg2cpu::default_model_dir().map(|d| d.join("tokenizer.json"));
    match path {
        Ok(p) if p.is_file() => Some(p),
        _ if std::env::var_os("EG2_REQUIRE_MODEL").is_some() => panic!("tokenizer_parity: no tokenizer.json"),
        _ => {
            eprintln!("SKIPPED tokenizer_parity: no checkpoint found");
            None
        }
    }
}

fn strings_in(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| strings_in(x, out)),
        Value::Object(o) => o.values().for_each(|x| strings_in(x, out)),
        _ => {}
    }
}

/// Source and docs under `dir` (recursively), skipping build output and fixtures.
fn text_files(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let name = p.file_name().unwrap().to_string_lossy();
        if p.is_dir() {
            if !matches!(name.as_ref(), "target" | ".git" | "node_modules" | "fixtures") {
                text_files(&p, out);
            }
        } else if p.extension().is_some_and(|x| ["rs", "md", "py", "pyi", "toml"].contains(&x.to_str().unwrap()))
            && let Ok(s) = std::fs::read_to_string(&p)
        {
            out.push(s);
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_text(rng: &mut Rng) -> String {
    const PIECES: &[&str] = &[
        "the",
        " the",
        "agent",
        " called",
        " read_file",
        ".",
        ",",
        "!",
        "?",
        "'",
        "\"",
        "(",
        ")",
        "{",
        "}",
        "::",
        "->",
        "0",
        "42",
        "3.14159",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\n\n",
        "\r\n",
        "\u{a0}",
        "\u{2003}",
        "▁",
        "▁▁",
        "é",
        "e\u{301}",
        "ß",
        "Ñandú",
        "中文",
        "日本語の文章",
        "한국어",
        "العربية",
        "עברית",
        "हिन्दी",
        "ไทย",
        "Ελληνικά",
        "кириллица",
        "👍",
        "👍🏽",
        "🧑\u{200d}💻",
        "🇨🇦",
        "\u{0}",
        "\u{1}",
        "\u{7f}",
        "\u{200b}",
        "\u{feff}",
        "\u{fffd}",
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<mask>",
        "<|image>",
        "<|image|>",
        "<image|>",
        "<|audio>",
        "<audio|>",
        "<|turn>",
        "<turn|>",
        "<|\"|>",
        "<",
        ">",
        "<|",
        "|>",
        "<bo",
        "os>",
        "<<bos>>",
        "<|ima",
        "<|image|",
        "</s>",
        "<start_of_turn>",
        "<0x0A>",
        "http://example.com/a?b=c&d=e",
        "fn main() {\n    println!(\"hi\");\n}",
    ];
    let mut s = String::new();
    for _ in 0..rng.below(120) {
        if rng.below(8) == 0 {
            // A random code point from the BMP or the supplementary planes.
            let max = if rng.below(4) == 0 { 0x3_0000 } else { 0x1_0000 };
            if let Some(c) = char::from_u32(rng.below(max) as u32) {
                s.push(c);
            }
        } else {
            s.push_str(PIECES[rng.below(PIECES.len())]);
        }
    }
    s
}

#[test]
fn matches_hugging_face_tokenizers() {
    let Some(path) = tokenizer_json() else { return };
    let t0 = Instant::now();
    let ours = Tokenizer::load(&path, MAX_TOKENS).unwrap();
    let load_ours = t0.elapsed();
    let t0 = Instant::now();
    let mut hf = tokenizers::Tokenizer::from_file(&path).unwrap();
    hf.with_truncation(Some(tokenizers::TruncationParams { max_length: MAX_TOKENS, ..Default::default() })).unwrap();
    hf.with_padding(None);
    let load_hf = t0.elapsed();

    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let mut texts = vec![String::new(), " ".into(), "  leading and trailing  ".into()];
    let mut files = Vec::new();
    text_files(Path::new(crate_dir), &mut files);
    text_files(&Path::new(crate_dir).join("../docs"), &mut files);
    for f in &files {
        texts.extend(f.split("\n\n").map(str::to_owned));
        texts.extend(f.lines().map(str::to_owned));
        texts.push(f.clone()); // whole files run past MAX_TOKENS
    }
    for e in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures")).unwrap().flatten() {
        if e.path().extension().is_some_and(|x| x == "json") {
            strings_in(&serde_json::from_slice(&std::fs::read(e.path()).unwrap()).unwrap(), &mut texts);
        }
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    texts.extend((0..5_000).map(|_| random_text(&mut rng)));
    texts.push((0..3000).map(|_| random_text(&mut rng)).collect());

    let (mut ours_time, mut hf_time) = (Duration::ZERO, Duration::ZERO);
    let mut truncated = 0;
    for text in &texts {
        let t0 = Instant::now();
        let got = ours.encode(text);
        ours_time += t0.elapsed();
        let t0 = Instant::now();
        let want = hf.encode(text.as_str(), true).unwrap().get_ids().to_vec();
        hf_time += t0.elapsed();
        assert_eq!(got, want, "ids differ for {text:?}");
        truncated += (got.len() == MAX_TOKENS) as usize;
    }
    println!(
        "{} texts ({truncated} truncated): eg2cpu load {load_ours:.2?} encode {ours_time:.2?}, \
         tokenizers load {load_hf:.2?} encode {hf_time:.2?}",
        texts.len()
    );
    assert!(truncated > 0, "no text reached MAX_TOKENS");
}
