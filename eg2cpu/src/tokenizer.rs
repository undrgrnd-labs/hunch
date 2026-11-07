//! The Gemma tokenizer, read from the checkpoint's `tokenizer.json`: special tokens are cut out of the raw text,
//! spaces in the rest become `▁`, each remaining span is one BPE word with byte fallback, and `<bos>`/`<eos>` go
//! around the result. Only the parts of the Hugging Face `tokenizers` format that EmbeddingGemma 2 uses are
//! implemented, and `load` refuses a file that asks for anything else. `tests/tokenizer_parity.rs` checks the ids
//! against the `tokenizers` crate.

use std::borrow::Cow;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};

/// (left id << 32 | right id) → (rank, merged id), from the `merges` list in order.
type Merges = HashMap<u64, (u32, u32), BuildHasherDefault<PairHash>>;

pub struct Tokenizer {
    merges: Merges,
    /// Tokens that are one character: a word starts as these before any merge.
    chars: HashMap<char, u32>,
    /// `<0xNN>` for each byte, used for characters without a token of their own (when byte fallback is on).
    bytes: [Option<u32>; 256],
    byte_fallback: bool,
    unk: Option<u32>,
    fuse_unk: bool,
    /// Special tokens, matched leftmost-longest in the raw text before anything else.
    specials: Vec<(Box<str>, u32)>,
    /// Whether some special token starts with this byte.
    special_first: [bool; 256],
    bos: u32,
    eos: u32,
    max_len: usize,
}

impl Tokenizer {
    /// Encodings are cut to `max_len` ids, `<bos>` and `<eos>` included.
    pub fn load(path: &Path, max_len: usize) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Mapped rather than read, so the 31 MB of JSON are clean file pages that never count against the process.
        // SAFETY: the checkpoint is read-only and not modified while mapped.
        let json = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_slice(&json, max_len).with_context(|| format!("tokenizer {}", path.display()))
    }

    pub fn from_slice(json: &[u8], max_len: usize) -> Result<Self> {
        ensure!(max_len >= 2, "max_len {max_len} leaves no room for <bos> and <eos>");
        let file: File = serde_json::from_slice(json)?;
        let model = file.model.ok_or_else(|| anyhow!("no model"))?;

        // Spaces become ▁ and nothing else changes; the split on " " afterwards then never matches.
        ensure!(
            file.normalizer == json!({"type": "Replace", "pattern": {"String": " "}, "content": "▁"}),
            "unsupported normalizer {}",
            file.normalizer
        );
        let pre = &file.pre_tokenizer;
        ensure!(
            pre.is_null()
                || pre["type"] == "Split" && pre["pattern"] == json!({"String": " "}) && pre["invert"] == false,
            "unsupported pre_tokenizer {pre}"
        );

        let opt = |k: &str| model.options.get(k).unwrap_or(&Value::Null);
        ensure!(opt("type") == "BPE", "unsupported model type {}", opt("type"));
        for k in ["dropout", "continuing_subword_prefix", "end_of_word_suffix"] {
            ensure!(opt(k).is_null(), "unsupported model.{k} {}", opt(k));
        }
        ensure!(opt("ignore_merges").as_bool() != Some(true), "unsupported model.ignore_merges");
        let vocab = model.vocab;
        let id = |t: &str| vocab.get(t).copied().ok_or_else(|| anyhow!("{t:?} is not in the vocabulary"));
        let unk = opt("unk_token").as_str().map(id).transpose()?;

        let chars = vocab
            .iter()
            .filter_map(|(t, &i)| {
                let mut cs = t.chars();
                cs.next().filter(|_| cs.next().is_none()).map(|c| (c, i))
            })
            .collect();
        let bytes = std::array::from_fn(|b| vocab.get(format!("<0x{b:02X}>").as_str()).copied());

        let mut specials = Vec::new();
        let mut special_first = [false; 256];
        for t in &file.added_tokens {
            let content = t["content"].as_str().ok_or_else(|| anyhow!("added token without content: {t}"))?;
            let tid = t["id"].as_u64().ok_or_else(|| anyhow!("added token without id: {t}"))? as u32;
            for k in ["normalized", "lstrip", "rstrip", "single_word"] {
                ensure!(t[k] == false, "unsupported added token {t}: {k}");
            }
            ensure!(!content.is_empty(), "empty added token");
            special_first[content.as_bytes()[0] as usize] = true;
            specials.push((content.into(), tid));
        }

        // `<bos>` A `<eos>`: the ids come from the template's own special_tokens table.
        let post = &file.post_processor;
        let special = |v: &Value| -> Result<u32> {
            let name = v["SpecialToken"]["id"].as_str().ok_or_else(|| anyhow!("expected a special token, got {v}"))?;
            match post["special_tokens"][name]["ids"].as_array().map(Vec::as_slice) {
                Some([i]) => i.as_u64().map(|i| i as u32).ok_or_else(|| anyhow!("bad id for {name}")),
                _ => bail!("special token {name} is not one id"),
            }
        };
        let (bos, eos) = match (post["type"].as_str(), post["single"].as_array().map(Vec::as_slice)) {
            (Some("TemplateProcessing"), Some([b, a, e])) if a == &json!({"Sequence": {"id": "A", "type_id": 0}}) => {
                (special(b)?, special(e)?)
            }
            _ => bail!("unsupported post_processor {post}"),
        };

        Ok(Self {
            merges: model.merges,
            chars,
            bytes,
            byte_fallback: opt("byte_fallback").as_bool().unwrap_or(false),
            unk,
            fuse_unk: opt("fuse_unk").as_bool().unwrap_or(false),
            specials,
            special_first,
            bos,
            eos,
            max_len,
        })
    }

    /// Token ids with `<bos>` and `<eos>`, cut to `max_len` (as `tokenizers` with right truncation).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.bos];
        let mut rest = text;
        let mut word = String::new();
        loop {
            let found = self.next_special(rest);
            let (span, after) = match found {
                Some((at, len, id)) => (&rest[..at], Some((&rest[at + len..], id))),
                None => (rest, None),
            };
            if !span.is_empty() {
                word.clear();
                for c in span.chars() {
                    word.push(if c == ' ' { '▁' } else { c });
                }
                self.bpe(&word, &mut ids);
            }
            match after {
                Some((r, id)) => {
                    ids.push(id);
                    rest = r;
                }
                None => break,
            }
            if ids.len() >= self.max_len - 1 {
                break;
            }
        }
        ids.truncate(self.max_len - 1);
        ids.push(self.eos);
        ids
    }

    /// The leftmost special token in `s`, the longest one there: (byte offset, length, id).
    fn next_special(&self, s: &str) -> Option<(usize, usize, u32)> {
        let b = s.as_bytes();
        (0..b.len()).filter(|&i| self.special_first[b[i] as usize]).find_map(|i| {
            self.specials
                .iter()
                .filter(|(t, _)| b[i..].starts_with(t.as_bytes()))
                .max_by_key(|(t, _)| t.len())
                .map(|(t, id)| (i, t.len(), *id))
        })
    }

    fn merge(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        self.merges.get(&pair(a, b)).copied()
    }

    /// One word: its characters (or their bytes, or `<unk>`), then the lowest-ranked merge anywhere in the word,
    /// leftmost first on ties, until none applies. The order of events matches `tokenizers` exactly, including
    /// that a queued merge still applies if its position now holds a different pair with the same result.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = Vec::with_capacity(word.len());
        let mut unk_pending = false;
        for c in word.chars() {
            if let Some(&id) = self.chars.get(&c) {
                if unk_pending {
                    ids.extend(self.unk);
                    unk_pending = false;
                }
                ids.push(id);
                continue;
            }
            if self.byte_fallback {
                let mut buf = [0u8; 4];
                let utf8 = c.encode_utf8(&mut buf).as_bytes();
                if utf8.iter().all(|&b| self.bytes[b as usize].is_some()) {
                    ids.extend(utf8.iter().map(|&b| self.bytes[b as usize].unwrap()));
                    continue;
                }
            }
            if self.unk.is_some() {
                if unk_pending && !self.fuse_unk {
                    ids.extend(self.unk);
                }
                unk_pending = true;
            }
        }
        if unk_pending {
            ids.extend(self.unk);
        }

        let n = ids.len();
        // Doubly linked symbols; `prev`/`next` are indices, usize::MAX for none.
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut next: Vec<usize> = (1..=n).map(|i| if i == n { usize::MAX } else { i }).collect();
        let mut live = vec![true; n];
        // Min-heap on (rank, position).
        let mut queue: BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>> = (0..n.saturating_sub(1))
            .filter_map(|i| self.merge(ids[i], ids[i + 1]).map(|(r, m)| std::cmp::Reverse((r, i, m))))
            .collect();
        while let Some(std::cmp::Reverse((_, pos, merged))) = queue.pop() {
            let right = next[pos];
            if !live[pos] || right == usize::MAX {
                continue;
            }
            if self.merge(ids[pos], ids[right]).map(|m| m.1) != Some(merged) {
                continue; // stale: one side merged since this was queued
            }
            ids[pos] = merged;
            live[right] = false;
            next[pos] = next[right];
            if next[pos] != usize::MAX {
                prev[next[pos]] = pos;
            }
            if prev[pos] != usize::MAX
                && let Some((r, m)) = self.merge(ids[prev[pos]], ids[pos])
            {
                queue.push(std::cmp::Reverse((r, prev[pos], m)));
            }
            if next[pos] != usize::MAX
                && let Some((r, m)) = self.merge(ids[pos], ids[next[pos]])
            {
                queue.push(std::cmp::Reverse((r, pos, m)));
            }
        }
        out.extend(ids.iter().zip(&live).filter(|(_, l)| **l).map(|(i, _)| *i));
    }
}

fn pair(a: u32, b: u32) -> u64 {
    (a as u64) << 32 | b as u64
}

/// The merge table's keys are two token ids; one multiply, folded, spreads both into the bucket bits.
#[derive(Default)]
struct PairHash(u64);

impl Hasher for PairHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(self.0 ^ b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        let m = x as u128 * 0x9e37_79b9_7f4a_7c15u128;
        self.0 = (m as u64) ^ (m >> 64) as u64;
    }
}

// ---------------------------------------------------------------- tokenizer.json

/// The parts of `tokenizer.json` that matter, read without building a `Value` for the 31 MB vocabulary and merges.
/// Token strings borrow from the JSON buffer unless they contain escapes, so loading allocates a few large tables
/// rather than a million small strings, and the merges go straight into their table. (macOS keeps large freed
/// blocks dirty, so a transient list of merges would stay in the process's footprint.)
#[derive(Default)]
struct File<'a> {
    added_tokens: Vec<Value>,
    normalizer: Value,
    pre_tokenizer: Value,
    post_processor: Value,
    model: Option<Bpe<'a>>,
}

struct Bpe<'a> {
    /// Everything in `model` but the vocabulary and merges.
    options: Map<String, Value>,
    vocab: HashMap<Cow<'a, str>, u32>,
    merges: Merges,
}

impl<'de> Deserialize<'de> for File<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = File<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a tokenizer.json object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<File<'de>, A::Error> {
                let mut file = File::default();
                while let Some(key) = map.next_key::<Str>()? {
                    match key.0.as_ref() {
                        "added_tokens" => file.added_tokens = map.next_value()?,
                        "normalizer" => file.normalizer = map.next_value()?,
                        "pre_tokenizer" => file.pre_tokenizer = map.next_value()?,
                        "post_processor" => file.post_processor = map.next_value()?,
                        "model" => file.model = Some(map.next_value()?),
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(file)
            }
        }
        d.deserialize_map(V)
    }
}

impl<'de> Deserialize<'de> for Bpe<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Bpe<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a BPE model object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Bpe<'de>, A::Error> {
                let mut bpe = Bpe { options: Map::new(), vocab: HashMap::new(), merges: Merges::default() };
                while let Some(key) = map.next_key::<Str>()? {
                    match key.0.as_ref() {
                        "vocab" => bpe.vocab = map.next_value::<Vocab>()?.0,
                        // `tokenizers` writes vocab first, which lets the merges resolve as they are read.
                        "merges" if bpe.vocab.is_empty() => return Err(de::Error::custom("merges before vocab")),
                        "merges" => bpe.merges = map.next_value_seed(MergesSeed(&bpe.vocab))?,
                        _ => {
                            let v = map.next_value()?;
                            bpe.options.insert(key.0.into_owned(), v);
                        }
                    }
                }
                Ok(bpe)
            }
        }
        d.deserialize_map(V)
    }
}

/// A JSON string, borrowed from the input when it has no escapes.
struct Str<'a>(Cow<'a, str>);

impl<'de> Deserialize<'de> for Str<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Str<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E: de::Error>(self, s: &'de str) -> Result<Str<'de>, E> {
                Ok(Str(Cow::Borrowed(s)))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Str<'de>, E> {
                Ok(Str(Cow::Owned(s.to_owned())))
            }
            fn visit_string<E: de::Error>(self, s: String) -> Result<Str<'de>, E> {
                Ok(Str(Cow::Owned(s)))
            }
        }
        d.deserialize_str(V)
    }
}

/// `{"token": id, ...}`
struct Vocab<'a>(HashMap<Cow<'a, str>, u32>);

impl<'de> Deserialize<'de> for Vocab<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vocab<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a vocabulary object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Vocab<'de>, A::Error> {
                let mut vocab = HashMap::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((k, v)) = map.next_entry::<Str, u32>()? {
                    vocab.insert(k.0, v);
                }
                Ok(Vocab(vocab))
            }
        }
        d.deserialize_map(V)
    }
}

/// The `merges` list, resolved to ids against the vocabulary as it is read.
struct MergesSeed<'v, 'a>(&'v HashMap<Cow<'a, str>, u32>);

impl<'de> DeserializeSeed<'de> for MergesSeed<'_, '_> {
    type Value = Merges;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Merges, D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for MergesSeed<'_, '_> {
    type Value = Merges;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a list of merges")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Merges, A::Error> {
        let id = |t: &str| {
            self.0
                .get(t)
                .copied()
                .ok_or_else(|| de::Error::custom(format!("merge token {t:?} is not in the vocabulary")))
        };
        // Gemma's list has about two merges per token; sizing for that avoids regrowing the table.
        let mut merges = Merges::with_capacity_and_hasher(2 * self.0.len(), Default::default());
        let mut joined = String::new();
        let mut rank = 0u32;
        while let Some(MergeEntry((a, b))) = seq.next_element()? {
            joined.clear();
            joined.push_str(&a);
            joined.push_str(&b);
            merges.insert(pair(id(&a)?, id(&b)?), (rank, id(&joined)?));
            rank += 1;
        }
        Ok(merges)
    }
}

/// A merge, written either `["a", "b"]` or `"a b"`.
struct MergeEntry<'a>((Cow<'a, str>, Cow<'a, str>));

impl<'de> Deserialize<'de> for MergeEntry<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        fn two<E: de::Error>(s: &str) -> Result<(&str, &str), E> {
            s.split_once(' ')
                .filter(|(_, b)| !b.contains(' '))
                .ok_or_else(|| E::custom(format!("merge {s:?} is not two tokens")))
        }
        impl<'de> Visitor<'de> for V {
            type Value = MergeEntry<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(r#"a merge, ["a", "b"] or "a b""#)
            }
            fn visit_borrowed_str<E: de::Error>(self, s: &'de str) -> Result<MergeEntry<'de>, E> {
                let (a, b) = two(s)?;
                Ok(MergeEntry((Cow::Borrowed(a), Cow::Borrowed(b))))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<MergeEntry<'de>, E> {
                let (a, b) = two(s)?;
                Ok(MergeEntry((Cow::Owned(a.into()), Cow::Owned(b.into()))))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<MergeEntry<'de>, A::Error> {
                let a: Str = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let b: Str = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                if seq.next_element::<IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(3, &self));
                }
                Ok(MergeEntry((a.0, b.0)))
            }
        }
        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer.json with the same structure as Gemma's: tokens a, b, ab, ▁, ▁ab, byte fallback, and specials.
    fn tiny(normalizer: &str, merges: &str) -> String {
        let mut vocab = Map::new();
        for (i, t) in ["<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "ab", "▁", "▁ab", "<|image>"].iter().enumerate()
        {
            vocab.insert(t.to_string(), json!(i));
        }
        for b in 0..256 {
            vocab.insert(format!("<0x{b:02X}>"), json!(10 + b));
        }
        let added: Vec<Value> = [(0, "<pad>"), (1, "<eos>"), (2, "<bos>"), (9, "<|image>")]
            .iter()
            .map(|(id, c)| {
                json!({"id": id, "content": c, "single_word": false, "lstrip": false, "rstrip": false,
                       "normalized": false, "special": true})
            })
            .collect();
        let special = |name: &str, id: u32| json!({"id": name, "ids": [id], "tokens": [name]});
        let post = json!({
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
            "special_tokens": {"<bos>": special("<bos>", 2), "<eos>": special("<eos>", 1)},
        });
        format!(
            r#"{{"version": "1.0", "added_tokens": {}, "normalizer": {normalizer},
                "pre_tokenizer": {{"type": "Split", "pattern": {{"String": " "}}, "behavior": "MergedWithPrevious",
                                   "invert": false}},
                "post_processor": {post}, "decoder": null,
                "model": {{"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
                           "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true,
                           "ignore_merges": false, "vocab": {}, "merges": {merges}}}}}"#,
            Value::from(added),
            Value::from(vocab),
        )
    }

    const SPACES: &str = r#"{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}"#;

    #[test]
    fn encodes_like_gemma() {
        for merges in [r#"[["a", "b"], ["▁", "ab"]]"#, r#"["a b", "▁ ab"]"#] {
            let t = Tokenizer::from_slice(tiny(SPACES, merges).as_bytes(), 1024).unwrap();
            assert_eq!(t.encode(""), [2, 1]);
            assert_eq!(t.encode(" ab"), [2, 8, 1]);
            assert_eq!(t.encode("aab b"), [2, 4, 6, 7, 5, 1]);
            // A broken special token is plain text: characters without a token of their own become bytes.
            let byte = |c: char| 10 + c as u32;
            let want = [2, 4, 9, 5, byte('<'), byte('|'), byte('i'), byte('m'), 4, byte('g'), byte('e'), 1];
            assert_eq!(t.encode("a<|image>b<|image"), want);
            assert_eq!(t.encode("é"), [2, 10 + 0xc3, 10 + 0xa9, 1]);
            let short = Tokenizer::from_slice(tiny(SPACES, merges).as_bytes(), 4).unwrap();
            assert_eq!(short.encode("abab ab"), [2, 6, 6, 1]);
        }
    }

    #[test]
    fn refuses_what_it_does_not_implement() {
        let merges = r#"[["a", "b"]]"#;
        let nfkc = Tokenizer::from_slice(tiny(r#"{"type": "NFKC"}"#, merges).as_bytes(), 1024);
        assert!(format!("{:#}", nfkc.err().unwrap()).contains("normalizer"));
        let missing = Tokenizer::from_slice(tiny(SPACES, r#"[["a", "z"]]"#).as_bytes(), 1024);
        assert!(format!("{:#}", missing.err().unwrap()).contains("not in the vocabulary"));
        let dropout = tiny(SPACES, merges).replace(r#""dropout": null"#, r#""dropout": 0.1"#);
        assert!(Tokenizer::from_slice(dropout.as_bytes(), 1024).is_err());
    }
}
