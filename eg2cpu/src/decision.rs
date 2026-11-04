//! Compiled contrastive decisions ("handles"): load one, gate and render a span, and apply it.
//!
//! A handle file is safetensors. Its f32 tensors are `directions` [k, dim] and, when the decision can abstain,
//! `abstain_mean` [dim] and `abstain_whiten` [dim, dim]. Everything else (kind, unit template, gate, readout
//! layer, calibration, escalation) is JSON in the `__metadata__` entry `handle`. The Python compiler
//! (`eg2cpu.decisions`) writes these files; this module only applies them.
//!
//! Applying many handles to a span costs one encoder pass per distinct rendered text, plus a dot product per
//! direction and (when abstaining is enabled) one dim x dim whitening product per handle.

use crate::simd::gemm_serial;
use crate::tensors::Tensors;
use crate::{Model, Output};
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// A unit of a trace to decide about: field name -> text (e.g. kind, call, output).
pub type Span = HashMap<String, String>;

pub const FORMAT: &str = "eg2-decision/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Boolean,
    Choice,
    Score,
    /// Ordered levels: one direction, proportional-odds cut-points.
    Levels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readout {
    /// The mean-pooled, normalized 768-d embedding.
    Final,
    /// The mean-pooled residual stream after layer i (0-based).
    Layer(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The gate rejected the span, or the unit template needs a field the span lacks.
    NotApplicable,
    /// The span is far from everything the decision was compiled and calibrated on.
    Abstain,
    /// In the uncertainty band: ask the escalation model with `prompt`.
    Escalate,
    Answer,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::NotApplicable => "not_applicable",
            Status::Abstain => "abstain",
            Status::Escalate => "escalate",
            Status::Answer => "answer",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub status: Status,
    /// Best guess even when abstaining or escalating: "yes"/"no" for boolean, the option label for choice, the most
    /// probable level's label for levels, None for score.
    pub answer: Option<String>,
    /// Calibrated probabilities: [p_yes] for boolean, one per label for choice and for levels (in level order),
    /// empty for score.
    pub p: Vec<f32>,
    /// Calibrated value for score decisions; for levels, the expected level index (sum of i * p[i]). None for
    /// boolean and choice.
    pub value: Option<f32>,
    /// Standardized readout scores, one per direction.
    pub z: Vec<f32>,
    /// Mahalanobis distance from the compile-time reference set, when the handle can abstain.
    pub distance: Option<f32>,
    /// The rendered escalation prompt, when `status` is `Escalate`.
    pub prompt: Option<String>,
}

impl Verdict {
    fn not_applicable() -> Self {
        Self {
            status: Status::NotApplicable,
            answer: None,
            p: vec![],
            value: None,
            z: vec![],
            distance: None,
            prompt: None,
        }
    }
}

// ---------------------------------------------------------------- templates

#[derive(Clone, Debug)]
enum Part {
    Lit(String),
    Field(String),
}

/// `{field}` placeholders over span fields; `{{` and `}}` are literal braces.
#[derive(Clone, Debug)]
pub struct Template(Vec<Part>);

impl Template {
    pub fn parse(s: &str) -> Result<Self> {
        let mut parts = Vec::new();
        let mut lit = String::new();
        let mut it = s.chars().peekable();
        while let Some(c) = it.next() {
            match c {
                '{' if it.peek() == Some(&'{') => {
                    it.next();
                    lit.push('{');
                }
                '}' if it.peek() == Some(&'}') => {
                    it.next();
                    lit.push('}');
                }
                '{' => {
                    let mut name = String::new();
                    loop {
                        match it.next() {
                            Some('}') => break,
                            Some(c) => name.push(c),
                            None => bail!("unclosed '{{' in template {s:?}"),
                        }
                    }
                    let name = name.trim().trim_start_matches("span.").to_string();
                    ensure!(!name.is_empty(), "empty placeholder in template {s:?}");
                    if !lit.is_empty() {
                        parts.push(Part::Lit(std::mem::take(&mut lit)));
                    }
                    parts.push(Part::Field(name));
                }
                '}' => bail!("unmatched '}}' in template {s:?}"),
                c => lit.push(c),
            }
        }
        if !lit.is_empty() {
            parts.push(Part::Lit(lit));
        }
        Ok(Self(parts))
    }

    /// The span fields the template names, in order.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.0.iter().filter_map(|p| match p {
            Part::Field(f) => Some(f.as_str()),
            Part::Lit(_) => None,
        })
    }

    /// None when the span lacks a field the template needs.
    pub fn render(&self, span: &Span, extra: Option<(&str, &str)>) -> Option<String> {
        let mut out = String::new();
        for p in &self.0 {
            match p {
                Part::Lit(s) => out.push_str(s),
                Part::Field(f) => match extra {
                    Some((k, v)) if k == f => out.push_str(v),
                    _ => out.push_str(span.get(f)?),
                },
            }
        }
        Some(out)
    }
}

// ---------------------------------------------------------------- gate expressions

/// `applies_when`: comparisons over span fields joined by and / or / not, with parentheses.
/// `span.kind == "tool_result"`, `span.tool in ["bash", "edit"]`, `span.output contains "Error"`,
/// `not (span.tool != "bash" or span.exit_code == 0)`, `span.expected == span.actual`. Fields are strings; a
/// `span.<field>` on the right is that field's value, any other bare word is literal text. `true` and `false`
/// match in any case on either side ("True" from Python's `str` equals `true`). A missing field, on either side,
/// makes `==`, `in` and `contains` false and `!=`, `not in` true.
#[derive(Clone, Debug)]
pub enum Gate {
    And(Box<Gate>, Box<Gate>),
    Or(Box<Gate>, Box<Gate>),
    Not(Box<Gate>),
    Cmp { field: String, op: CmpOp, values: Vec<Operand> },
}

/// The right-hand side of a gate comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Lit(String),
    /// `span.<field>`: the span's value of the field.
    Field(String),
}

impl Operand {
    fn get<'a>(&'a self, span: &'a Span) -> Option<&'a str> {
        match self {
            Operand::Lit(s) => Some(s),
            Operand::Field(f) => span.get(f).map(String::as_str),
        }
    }
}

/// Booleans in any case compare as `true` / `false`; everything else as written.
fn same(a: &str, b: &str) -> bool {
    let bool_word = |s: &str| s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false");
    if bool_word(a) && bool_word(b) { a.eq_ignore_ascii_case(b) } else { a == b }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    In,
    NotIn,
    Contains,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Op(&'static str),
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '"' || c == '\'' {
            let mut v = String::new();
            i += 1;
            loop {
                ensure!(i < cs.len(), "unterminated string in {s:?}");
                match cs[i] {
                    '\\' if i + 1 < cs.len() => {
                        v.push(match cs[i + 1] {
                            'n' => '\n',
                            't' => '\t',
                            o => o,
                        });
                        i += 2;
                    }
                    q if q == c => {
                        i += 1;
                        break;
                    }
                    o => {
                        v.push(o);
                        i += 1;
                    }
                }
            }
            out.push(Tok::Str(v));
        } else if (c == '=' || c == '!') && cs.get(i + 1) == Some(&'=') {
            out.push(Tok::Op(if c == '=' { "==" } else { "!=" }));
            i += 2;
        } else if "()[],".contains(c) {
            out.push(Tok::Op(match c {
                '(' => "(",
                ')' => ")",
                '[' => "[",
                ']' => "]",
                _ => ",",
            }));
            i += 1;
        } else if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
            let start = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || "_.-".contains(cs[i])) {
                i += 1;
            }
            let w: String = cs[start..i].iter().collect();
            out.push(match w.as_str() {
                "and" => Tok::Op("and"),
                "or" => Tok::Op("or"),
                "not" => Tok::Op("not"),
                "in" => Tok::Op("in"),
                "contains" => Tok::Op("contains"),
                "true" | "True" => Tok::Str("true".into()),
                "false" | "False" => Tok::Str("false".into()),
                _ => Tok::Ident(w),
            });
        } else {
            bail!("unexpected {c:?} in gate {s:?}");
        }
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn eat(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, op: &str) -> Result<()> {
        ensure!(self.eat(op), "expected {op:?} at token {}", self.pos);
        Ok(())
    }
    fn or(&mut self) -> Result<Gate> {
        let mut g = self.and()?;
        while self.eat("or") {
            g = Gate::Or(Box::new(g), Box::new(self.and()?));
        }
        Ok(g)
    }
    fn and(&mut self) -> Result<Gate> {
        let mut g = self.not()?;
        while self.eat("and") {
            g = Gate::And(Box::new(g), Box::new(self.not()?));
        }
        Ok(g)
    }
    fn not(&mut self) -> Result<Gate> {
        if self.eat("not") {
            return Ok(Gate::Not(Box::new(self.not()?)));
        }
        if self.eat("(") {
            let g = self.or()?;
            self.expect(")")?;
            return Ok(g);
        }
        self.cmp()
    }
    fn value(&mut self) -> Result<Operand> {
        match self.toks.get(self.pos).cloned() {
            Some(Tok::Ident(s)) if s.starts_with("span.") => {
                self.pos += 1;
                Ok(Operand::Field(s.trim_start_matches("span.").to_string()))
            }
            Some(Tok::Str(s)) | Some(Tok::Ident(s)) => {
                self.pos += 1;
                Ok(Operand::Lit(s))
            }
            t => bail!("expected a value, got {t:?}"),
        }
    }
    fn cmp(&mut self) -> Result<Gate> {
        let field = match self.toks.get(self.pos).cloned() {
            Some(Tok::Ident(f)) => {
                self.pos += 1;
                f.trim_start_matches("span.").to_string()
            }
            t => bail!("expected a span field, got {t:?}"),
        };
        let op = if self.eat("==") {
            CmpOp::Eq
        } else if self.eat("!=") {
            CmpOp::Ne
        } else if self.eat("in") {
            CmpOp::In
        } else if self.eat("contains") {
            CmpOp::Contains
        } else if self.eat("not") {
            self.expect("in")?;
            CmpOp::NotIn
        } else {
            bail!("expected ==, !=, in, not in or contains after {field:?}");
        };
        let values = if matches!(op, CmpOp::In | CmpOp::NotIn) {
            self.expect("[")?;
            let mut v = Vec::new();
            if !self.eat("]") {
                loop {
                    v.push(self.value()?);
                    if self.eat("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            v
        } else {
            vec![self.value()?]
        };
        Ok(Gate::Cmp { field, op, values })
    }
}

impl Gate {
    pub fn parse(s: &str) -> Result<Self> {
        let mut p = Parser { toks: lex(s)?, pos: 0 };
        let g = p.or().with_context(|| format!("in gate {s:?}"))?;
        ensure!(p.pos == p.toks.len(), "trailing tokens in gate {s:?}");
        Ok(g)
    }

    pub fn eval(&self, span: &Span) -> bool {
        match self {
            Gate::And(a, b) => a.eval(span) && b.eval(span),
            Gate::Or(a, b) => a.eval(span) || b.eval(span),
            Gate::Not(a) => !a.eval(span),
            Gate::Cmp { field, op, values } => {
                let v = span.get(field).map(String::as_str);
                let found = || v.is_some_and(|v| values.iter().any(|x| x.get(span).is_some_and(|x| same(v, x))));
                match op {
                    CmpOp::Eq | CmpOp::In => found(),
                    CmpOp::Ne | CmpOp::NotIn => !found(),
                    CmpOp::Contains => v.zip(values[0].get(span)).is_some_and(|(v, x)| v.contains(x)),
                }
            }
        }
    }
}

// ---------------------------------------------------------------- handles

#[derive(Clone, Debug)]
enum Rule {
    /// |p_yes - center| < width (boolean) or |value - center| < width (score)
    Band { center: f32, width: f32 },
    /// top probability below the value (choice)
    PMaxBelow(f32),
    /// top minus runner-up probability below the value (choice)
    MarginBelow(f32),
}

#[derive(Clone, Debug)]
struct Escalation {
    rule: Rule,
    prompt: Template,
}

#[derive(Clone, Debug)]
struct Abstain {
    mean: Vec<f32>,
    /// dim x dim, row-major: distance = |W (x - mean)|
    whiten: Vec<f32>,
    threshold: f32,
}

#[derive(Clone, Debug)]
pub struct Handle {
    pub name: String,
    pub kind: Kind,
    /// Boolean: ["no", "yes"]; choice: the options; levels: the levels, lowest first; score: [].
    pub labels: Vec<String>,
    /// Encoder revision the handle was compiled against ("" accepts any).
    pub revision: String,
    pub readout: Readout,
    /// The full metadata JSON, as written by the compiler.
    pub meta: Value,
    unit: Template,
    gate: Option<Gate>,
    directions: Vec<Vec<f32>>,
    std_mean: Vec<f32>,
    std_sd: Vec<f32>,
    cal_w: Vec<Vec<f32>>,
    cal_b: Vec<f32>,
    abstain: Option<Abstain>,
    escalate: Option<Escalation>,
}

fn f32s(v: &Value, what: &str) -> Result<Vec<f32>> {
    v.as_array()
        .with_context(|| format!("{what}: expected a list"))?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|x| x as f32)
                .filter(|x| x.is_finite())
                .with_context(|| format!("{what}: expected finite numbers"))
        })
        .collect()
}

/// An f32 tensor's shape and values, `None` if the handle has no such tensor.
fn take(t: &Tensors, name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    t.shape(name).map(|shape| Ok((shape.to_vec(), t.f32(name, shape)?))).transpose()
}

impl Handle {
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
            .with_context(|| format!("loading handle {}", path.display()))
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let t = Tensors::parse(buf)?;
        for (name, dtype) in t.dtypes() {
            ensure!(dtype == "F32", "{name}: expected F32, got {dtype}");
        }
        let m: Value = serde_json::from_str(t.metadata().get("handle").context("no 'handle' metadata")?)?;
        ensure!(m["format"] == FORMAT, "unsupported handle format {:?}", m["format"]);
        let kind = match m["kind"].as_str() {
            Some("boolean") => Kind::Boolean,
            Some("choice") => Kind::Choice,
            Some("score") => Kind::Score,
            Some("levels") => Kind::Levels,
            k => bail!("unknown kind {k:?}"),
        };
        let labels: Vec<String> = m["labels"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let readout = match &m["readout"]["layer"] {
            Value::String(s) if s == "final" => Readout::Final,
            Value::Number(n) => Readout::Layer(n.as_u64().context("readout.layer")? as usize),
            l => bail!("readout.layer must be \"final\" or a layer index, got {l}"),
        };
        let (dshape, dflat) = take(&t, "directions")?.context("missing 'directions' tensor")?;
        ensure!(dshape.len() == 2 && dshape[0] > 0 && dshape[1] > 0, "directions must be [k, dim], both non-zero");
        ensure!(dflat.iter().all(|x| x.is_finite()), "directions must be finite");
        let (k, dim) = (dshape[0], dshape[1]);
        let directions: Vec<Vec<f32>> = dflat.chunks_exact(dim).map(<[f32]>::to_vec).collect();
        let std_mean = f32s(&m["standardize"]["mean"], "standardize.mean")?;
        let std_sd = f32s(&m["standardize"]["std"], "standardize.std")?;
        ensure!(std_mean.len() == k && std_sd.len() == k, "standardize needs one mean/std per direction");
        // A zero sd would make every z NaN, and a NaN probability reads as a confident "no".
        ensure!(std_sd.iter().all(|&s| s > 0.0), "standardize.std must be positive");
        let cal = &m["calibration"];
        let cal_w: Vec<Vec<f32>> = cal["weights"]
            .as_array()
            .context("calibration.weights")?
            .iter()
            .map(|r| f32s(r, "calibration.weights"))
            .collect::<Result<_>>()?;
        let cal_b = f32s(&cal["bias"], "calibration.bias")?;
        let (outputs, biases) = match kind {
            Kind::Choice => {
                ensure!(labels.len() == k, "choice needs one direction per label");
                (k, k)
            }
            Kind::Levels => {
                ensure!(k == 1 && labels.len() >= 2, "levels needs one direction and at least two levels");
                let ok = cal_b.windows(2).all(|w| w[0] <= w[1]);
                ensure!(ok, "levels cut-points must be increasing");
                (1, labels.len() - 1) // a slope, and one cut-point between each pair of adjacent levels
            }
            _ => {
                ensure!(k == 1, "{kind:?} needs exactly one direction");
                (1, 1)
            }
        };
        ensure!(
            cal_w.len() == outputs && cal_b.len() == biases && cal_w.iter().all(|r| r.len() == k),
            "calibration shape mismatch"
        );
        let abstain = match (&m["abstain"], take(&t, "abstain_mean")?, take(&t, "abstain_whiten")?) {
            (Value::Object(a), Some((_, mean)), Some((wshape, whiten))) => {
                ensure!(mean.len() == dim && wshape == [dim, dim], "abstain tensors must be [dim] and [dim, dim]");
                Some(Abstain { mean, whiten, threshold: a["threshold"].as_f64().context("abstain.threshold")? as f32 })
            }
            (Value::Null, ..) => None,
            _ => bail!("abstain metadata and tensors disagree"),
        };
        let unit = Template::parse(m["unit"]["template"].as_str().context("unit.template")?)?;
        let escalate = match &m["escalate"] {
            Value::Null => None,
            e => {
                let r = &e["rule"];
                let num = |k: &str| r[k].as_f64().map(|x| x as f32).with_context(|| format!("escalate.rule.{k}"));
                let rule = match r["type"].as_str() {
                    Some("band") => Rule::Band { center: num("center")?, width: num("width")? },
                    Some("p_max_below") => Rule::PMaxBelow(num("value")?),
                    Some("margin_below") => Rule::MarginBelow(num("value")?),
                    t => bail!("unknown escalate rule {t:?}"),
                };
                let fits = match rule {
                    Rule::Band { .. } => kind != Kind::Choice,
                    Rule::PMaxBelow(_) | Rule::MarginBelow(_) => matches!(kind, Kind::Choice | Kind::Levels),
                };
                ensure!(fits, "escalate rule {:?} does not apply to a {kind:?} decision", r["type"]);
                let prompt = Template::parse(e["prompt"].as_str().context("escalate.prompt")?)?;
                // Every field the prompt names must be one the unit has, so an escalate verdict always has a prompt.
                for f in prompt.fields() {
                    ensure!(
                        f == "unit" || unit.fields().any(|u| u == f),
                        "escalate.prompt names {{{f}}}, which the unit template lacks"
                    );
                }
                Some(Escalation { rule, prompt })
            }
        };
        let gate = match m["applies_when"].as_str() {
            Some(s) if !s.trim().is_empty() => Some(Gate::parse(s)?),
            _ => None,
        };
        Ok(Self {
            name: m["decision"].as_str().unwrap_or_default().to_string(),
            kind,
            labels,
            revision: m["encoder"]["revision"].as_str().unwrap_or_default().to_string(),
            readout,
            unit,
            gate,
            directions,
            std_mean,
            std_sd,
            cal_w,
            cal_b,
            abstain,
            escalate,
            meta: m,
        })
    }

    pub fn dim(&self) -> usize {
        self.directions[0].len()
    }

    /// The text to embed for this span, or None if the decision does not apply to it.
    pub fn render(&self, span: &Span) -> Option<String> {
        if self.gate.as_ref().is_some_and(|g| !g.eval(span)) {
            return None;
        }
        self.unit.render(span, None)
    }

    /// Score an embedded unit. `unit_text` is what `render` returned for this span.
    pub fn score(&self, out: &Output, span: &Span, unit_text: &str) -> Result<Verdict> {
        let x: &[f32] = match self.readout {
            Readout::Final => &out.embedding,
            Readout::Layer(i) => out.layer_means.get(i).with_context(|| format!("{}: no layer {i}", self.name))?,
        };
        ensure!(x.len() == self.dim(), "{}: readout has {} dims, handle expects {}", self.name, x.len(), self.dim());
        let z: Vec<f32> = self
            .directions
            .iter()
            .zip(self.std_mean.iter().zip(&self.std_sd))
            .map(|(w, (m, s))| (dot(w, x) - m) / s)
            .collect();
        let logits: Vec<f32> = self.cal_w.iter().zip(&self.cal_b).map(|(w, b)| dot(w, &z) + b).collect();
        let (answer, p, value) = match self.kind {
            Kind::Levels => {
                // Proportional odds: P(level <= k) = sigmoid(theta_k - a z).
                let eta = self.cal_w[0][0] * z[0];
                let cum: Vec<f32> = self.cal_b.iter().map(|t| sigmoid(t - eta)).chain([1.0]).collect();
                let p: Vec<f32> =
                    cum.iter().enumerate().map(|(i, c)| c - if i == 0 { 0.0 } else { cum[i - 1] }).collect();
                let expected = p.iter().enumerate().map(|(i, q)| i as f32 * q).sum();
                (Some(self.labels[argmax(&p)].clone()), p, Some(expected))
            }
            Kind::Boolean => {
                let p = sigmoid(logits[0]);
                (Some(if p >= 0.5 { "yes" } else { "no" }.to_string()), vec![p], None)
            }
            Kind::Choice => {
                let p = softmax(&logits);
                let best = argmax(&p);
                (Some(self.labels[best].clone()), p, None)
            }
            Kind::Score => (None, vec![], Some(logits[0])),
        };
        let distance = self.abstain.as_ref().map(|a| {
            let d: Vec<f32> = x.iter().zip(&a.mean).map(|(x, m)| x - m).collect();
            let mut y = vec![0.0f32; d.len()];
            gemm_serial(1, d.len(), d.len(), &d, d.len(), &a.whiten, d.len(), &mut y, d.len(), false);
            y.iter().map(|v| v * v).sum::<f32>().sqrt()
        });
        let far = matches!((&self.abstain, distance), (Some(a), Some(d)) if d > a.threshold);
        let uncertain = self.escalate.as_ref().is_some_and(|e| match e.rule {
            Rule::Band { center, width } => match self.kind {
                Kind::Score | Kind::Levels => value.is_some_and(|v| (v - center).abs() < width),
                _ => (p[0] - center).abs() < width,
            },
            Rule::PMaxBelow(t) => p.iter().cloned().fold(0.0, f32::max) < t,
            Rule::MarginBelow(t) => {
                let mut s = p.clone();
                s.sort_by(|a, b| b.total_cmp(a));
                s[0] - s.get(1).copied().unwrap_or(0.0) < t
            }
        });
        let status = if far {
            Status::Abstain
        } else if uncertain {
            Status::Escalate
        } else {
            Status::Answer
        };
        let prompt = match (status, &self.escalate) {
            (Status::Escalate, Some(e)) => e.prompt.render(span, Some(("unit", unit_text))),
            _ => None,
        };
        Ok(Verdict { status, answer, p, value, z, distance, prompt })
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

impl Model {
    /// Apply every handle to every span. Each distinct rendered unit text is encoded once, in one concurrent
    /// batch; the result is indexed [span][handle].
    pub fn apply(&self, handles: &[&Handle], spans: &[Span]) -> Result<Vec<Vec<Verdict>>> {
        for h in handles {
            ensure!(
                h.revision.is_empty() || h.revision == self.revision(),
                "handle {:?} was compiled against encoder revision {}, but the loaded model is {}",
                h.name,
                h.revision,
                self.revision()
            );
        }
        let mut texts: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let plan: Vec<Vec<Option<usize>>> = spans
            .iter()
            .map(|span| {
                handles
                    .iter()
                    .map(|h| {
                        h.render(span).map(|t| {
                            *index.entry(t.clone()).or_insert_with(|| {
                                texts.push(t);
                                texts.len() - 1
                            })
                        })
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let outs = self.embed_batch(&refs)?;
        plan.iter()
            .zip(spans)
            .map(|(row, span)| {
                row.iter()
                    .zip(handles)
                    .map(|(ti, h)| match ti {
                        Some(i) => h.score(&outs[*i], span, &texts[*i]),
                        None => Ok(Verdict::not_applicable()),
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kv: &[(&str, &str)]) -> Span {
        kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn gates() {
        let s = span(&[("kind", "tool_result"), ("tool", "bash"), ("output", "Error: boom")]);
        let t = |g: &str| Gate::parse(g).unwrap().eval(&s);
        assert!(t(r#"span.kind == "tool_result""#));
        assert!(!t(r#"span.kind != 'tool_result'"#));
        assert!(t(r#"span.tool in ["bash", "edit"] and span.output contains "Error""#));
        assert!(t(r#"span.tool not in [read] or span.missing == "x""#));
        assert!(t(r#"not (span.tool == "read" or span.missing == "x")"#));
        assert!(t(r#"span.missing != "x" and span.missing not in ["a"]"#));
        assert!(!t(r#"span.missing contains "x""#));
        for bad in
            ["span.kind ==", "span.kind = 'x'", "(span.kind == 'x'", "span.kind in 'x'", "span.kind == 'x' extra"]
        {
            assert!(Gate::parse(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn templates() {
        let s = span(&[("call", "bash: ls"), ("output", "a b")]);
        assert_eq!(Template::parse("{call}\n{output}").unwrap().render(&s, None).as_deref(), Some("bash: ls\na b"));
        assert_eq!(
            Template::parse("{{literal}} {span.call}").unwrap().render(&s, None).as_deref(),
            Some("{literal} bash: ls")
        );
        assert_eq!(Template::parse("{unit}!").unwrap().render(&s, Some(("unit", "u"))).as_deref(), Some("u!"));
        assert!(Template::parse("{text}").unwrap().render(&s, None).is_none());
        assert!(Template::parse("{call").is_err());
        assert_eq!(Template::parse("{{x}} {unit} {span.call}").unwrap().fields().collect::<Vec<_>>(), ["unit", "call"]);
    }

    #[test]
    fn gate_operands() {
        let s = span(&[("a", "x"), ("b", "x"), ("c", "y"), ("flag", "True"), ("off", "FALSE")]);
        let t = |g: &str| Gate::parse(g).unwrap().eval(&s);
        // span.<field> on the right is the field's value, not the text "span.b"
        assert!(t("span.a == span.b") && !t("span.a == span.c") && t("span.a != span.c"));
        assert!(t("span.a in [span.c, span.b]") && t("span.a not in [span.c]") && t("span.a contains span.b"));
        assert!(!t("span.a == span.missing") && t("span.a != span.missing") && !t("span.a contains span.missing"));
        assert!(t("span.a == x") && !t("span.a == span.x"));
        // booleans match in any case: Python's str(True) is "True"
        assert!(
            t("span.flag == true") && t(r#"span.flag == "true""#) && t("span.flag == True") && !t("span.flag == false")
        );
        assert!(t("span.off == false") && t("span.flag in [true]") && !t("span.flag != true"));
        assert!(t("span.flag == span.flag") && !t(r#"span.a == "X""#));
    }

    /// A handle file from metadata and named f32 tensors.
    fn handle_bytes(meta: &Value, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::json!({"__metadata__": {"handle": meta.to_string()}});
        let (mut data, mut off) = (Vec::new(), 0);
        for (name, shape, v) in tensors {
            header[*name] =
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + 4 * v.len()]});
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            off += 4 * v.len();
        }
        let h = header.to_string().into_bytes();
        [&(h.len() as u64).to_le_bytes()[..], &h, &data].concat()
    }

    fn boolean_meta() -> Value {
        serde_json::json!({
            "format": FORMAT, "decision": "t", "kind": "boolean", "labels": ["no", "yes"],
            "unit": {"template": "{call}\n{output}"}, "applies_when": "", "encoder": {"revision": ""},
            "readout": {"layer": "final"}, "standardize": {"mean": [0.5], "std": [2.0]},
            "calibration": {"weights": [[1.5]], "bias": [-0.25]}, "abstain": null,
            "escalate": {"rule": {"type": "band", "center": 0.5, "width": 0.2}, "prompt": "{unit}\n\nDid it fail?"},
        })
    }

    fn load(meta: &Value, dshape: Vec<usize>) -> Result<Handle> {
        let n = dshape.iter().product();
        Handle::from_bytes(&handle_bytes(meta, &[("directions", dshape, vec![0.25; n])]))
    }

    fn err(r: Result<Handle>) -> String {
        format!("{:#}", r.expect_err("should not load"))
    }

    #[test]
    fn malformed_handles_are_errors() {
        let good = handle_bytes(&boolean_meta(), &[("directions", vec![1, 4], vec![0.25; 4])]);
        assert!(Handle::from_bytes(&good).is_ok());
        // Every truncation and every single-byte change of a good file loads or errors; none panics.
        for n in 0..good.len() {
            let _ = Handle::from_bytes(&good[..n]);
        }
        for i in 0..good.len() {
            for b in [b'0', b'9', b'-', b'"', b'[', 0xff] {
                let mut bad = good.clone();
                bad[i] = b;
                let _ = Handle::from_bytes(&bad);
            }
        }
        let with_header = |h: &str| [&(h.len() as u64).to_le_bytes()[..], h.as_bytes()].concat();
        assert!(Handle::from_bytes(&u64::MAX.to_le_bytes()).is_err());
        for t in [
            r#"{"directions":{"dtype":"F32","shape":[1,1],"data_offsets":[0,18446744073709551615]}}"#,
            r#"{"directions":{"dtype":"F32","shape":[4294967296,4294967296,4],"data_offsets":[0,0]}}"#,
            r#"{"directions":{"dtype":"F32","shape":[1,1],"data_offsets":[4,0]}}"#,
        ] {
            assert!(err(Handle::from_bytes(&with_header(t))).contains("directions: "), "{t}");
        }
        assert!(err(load(&boolean_meta(), vec![1, 0])).contains("non-zero"));
        assert!(err(load(&boolean_meta(), vec![1])).contains("[k, dim]"));
        let mut m = boolean_meta();
        m["standardize"]["std"] = serde_json::json!([0.0]);
        assert!(err(load(&m, vec![1, 4])).contains("std must be positive"));
        // the loader stays strict: NaN is not JSON
        let nan = handle_bytes(&boolean_meta(), &[("directions", vec![1, 4], vec![0.25; 4])]);
        let nan = String::from_utf8_lossy(&nan).replace("[-0.25]", "[NaN]  ").into_bytes();
        assert!(Handle::from_bytes(&nan).is_err());
    }

    #[test]
    fn escalation_must_fit_the_handle() {
        let mut m = boolean_meta();
        m["kind"] = "score".into();
        m["labels"] = serde_json::json!([]);
        m["escalate"]["rule"] = serde_json::json!({"type": "margin_below", "value": 0.1});
        assert!(err(load(&m, vec![1, 4])).contains("does not apply"));
        let mut m = boolean_meta();
        m["escalate"]["prompt"] = "Tool {tool} returned {output}. Did it fail?".into();
        assert!(err(load(&m, vec![1, 4])).contains("{tool}"));
        m["escalate"]["prompt"] = "Call {call} returned {output}. Did it fail? {unit}".into();
        assert!(load(&m, vec![1, 4]).is_ok());
    }
}
