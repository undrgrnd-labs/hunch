//! eg2 embed <text>...                          print one JSON embedding per line
//! eg2 bench [tokens...]                        time forward passes at the given lengths (default 16 128 512 1024),
//!                                              then batch throughput for 256 texts of 32 and 128 tokens
//! eg2 batch <texts> <tokens> <in-flight>...    forward_batch_with throughput for each in-flight bound
//! eg2 callers <threads> <reqs> <tokens>        independent single-text requests from OS threads, each sending <reqs>
//! eg2 mixed <threads> <reqs> <tokens>          the same callers while another thread runs a 512-text x 128-token batch
//! eg2 exit <layer> [tokens...]                 early exit after <layer> (layers 0..=layer, as `embed_batch_partial`)
//!                                              for one text and for 256 texts, against the full pass (tokens default
//!                                              to 32 128)
//! eg2 image <file> [budgets...]                time embedding one image at each soft-token budget (default 70 140
//!                                              280): decode + resize, vision tower, text tower, and the whole
//!                                              `embed_image_with`
//! eg2 image-embed <file>...                    print one JSON embedding per image (280 soft tokens)
//! eg2 audio <file.wav> [seconds...]            time embedding the clip cut (or repeated) to each length (default 5 10
//!                                              30): log-mel features, audio tower, text tower, and the whole
//!                                              `embed_audio_samples`
//! eg2 audio-embed <file.wav>...                print one JSON embedding per clip
//!
//! "cores busy" is process CPU time over wall time: how many cores the run kept busy on average.

use anyhow::{Context, Result, anyhow, bail, ensure};
use eg2cpu::{Model, SOFT_TOKEN_BUDGETS, isa};
use std::time::Instant;

const USAGE: &str = "usage: eg2 embed <text>... | eg2 bench [tokens...] | eg2 batch <texts> <tokens> <in-flight>... | \
                     eg2 callers <threads> <reqs> <tokens> | eg2 mixed <threads> <reqs> <tokens> | \
                     eg2 exit <layer> [tokens...] | eg2 image <file> [budgets...] | eg2 image-embed <file>... | \
                     eg2 audio <file.wav> [seconds...] | eg2 audio-embed <file.wav>...";

/// User + system CPU seconds used by this process so far (0 where getrusage isn't available).
fn cpu_secs() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        let t = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
        t(u.ru_utime) + t(u.ru_stime)
    }
    #[cfg(not(unix))]
    0.0
}

/// Value at quantile `f` of sorted `xs`.
fn quantile(xs: &[f64], f: f64) -> f64 {
    xs[((xs.len() - 1) as f64 * f) as usize]
}

/// `per` single-text requests of `len` tokens from each of `threads` callers; sorted latencies in ms.
fn callers(model: &Model, base: &[u32], threads: usize, per: usize, len: usize) -> Vec<f64> {
    let mut lat: Vec<f64> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|c| {
                s.spawn(move || {
                    (0..per)
                        .map(|r| {
                            let t = Instant::now();
                            std::hint::black_box(model.forward(&base[(c + r) % 7..(c + r) % 7 + len]));
                            t.elapsed().as_secs_f64() * 1e3
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    lat.sort_by(f64::total_cmp);
    lat
}

/// A subcommand and its arguments, parsed before the model loads.
enum Cmd {
    Embed(Vec<String>),
    Bench(Vec<usize>),
    Batch { texts: usize, len: usize, in_flight: Vec<usize> },
    Callers { threads: usize, per: usize, len: usize },
    Mixed { threads: usize, per: usize, len: usize },
    Exit { upto: usize, lens: Vec<usize> },
    Image { path: String, budgets: Vec<usize> },
    ImageEmbed(Vec<String>),
    Audio { path: String, seconds: Vec<usize> },
    AudioEmbed(Vec<String>),
}

fn number(what: &str, a: &str) -> Result<usize> {
    a.parse().with_context(|| format!("<{what}> must be a non-negative integer, got {a:?}"))
}

fn positive(what: &str, a: &str) -> Result<usize> {
    let n = number(what, a)?;
    ensure!(n > 0, "<{what}> must be positive");
    Ok(n)
}

/// Positional argument `i`.
fn arg<'a>(args: &'a [String], i: usize, what: &str) -> Result<&'a str> {
    args.get(i).map(String::as_str).with_context(|| format!("missing <{what}>"))
}

/// Arguments `i..`, at least one.
fn rest<'a>(args: &'a [String], i: usize, what: &str) -> Result<&'a [String]> {
    ensure!(args.len() > i, "missing <{what}>");
    Ok(&args[i..])
}

fn parse(args: &[String]) -> Result<Cmd> {
    let Some(sub) = args.first() else { bail!("missing subcommand") };
    let tokens_from = |i: usize, default: &[usize]| -> Result<Vec<usize>> {
        if args.len() > i { args[i..].iter().map(|a| positive("tokens", a)).collect() } else { Ok(default.to_vec()) }
    };
    Ok(match sub.as_str() {
        "embed" => Cmd::Embed(rest(args, 1, "text")?.to_vec()),
        "bench" => Cmd::Bench(tokens_from(1, &[16, 128, 512, 1024])?),
        "batch" => Cmd::Batch {
            texts: positive("texts", arg(args, 1, "texts")?)?,
            len: positive("tokens", arg(args, 2, "tokens")?)?,
            in_flight: rest(args, 3, "in-flight")?.iter().map(|a| number("in-flight", a)).collect::<Result<_>>()?,
        },
        "callers" | "mixed" => {
            let threads = positive("threads", arg(args, 1, "threads")?)?;
            let per = positive("reqs", arg(args, 2, "reqs")?)?;
            let len = positive("tokens", arg(args, 3, "tokens")?)?;
            if sub == "callers" { Cmd::Callers { threads, per, len } } else { Cmd::Mixed { threads, per, len } }
        }
        "exit" => Cmd::Exit { upto: number("layer", arg(args, 1, "layer")?)?, lens: tokens_from(2, &[32, 128])? },
        "image" => Cmd::Image {
            path: arg(args, 1, "file")?.to_string(),
            budgets: if args.len() > 2 {
                args[2..]
                    .iter()
                    .map(|a| {
                        let n = number("budgets", a)?;
                        ensure!(SOFT_TOKEN_BUDGETS.contains(&n), "<budgets> must be among {SOFT_TOKEN_BUDGETS:?}");
                        Ok(n)
                    })
                    .collect::<Result<_>>()?
            } else {
                vec![70, 140, 280]
            },
        },
        "image-embed" => Cmd::ImageEmbed(rest(args, 1, "file")?.to_vec()),
        "audio" => Cmd::Audio {
            path: arg(args, 1, "file")?.to_string(),
            seconds: if args.len() > 2 {
                args[2..].iter().map(|a| positive("seconds", a)).collect::<Result<_>>()?
            } else {
                vec![5, 10, 30]
            },
        },
        "audio-embed" => Cmd::AudioEmbed(rest(args, 1, "file")?.to_vec()),
        other => bail!("unknown subcommand {other:?}"),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = parse(&args).map_err(|e| anyhow!("{e:#}\n{USAGE}"))?;
    let t0 = Instant::now();
    let model = Model::load_default()?;
    eprintln!("loaded in {:.2?} (isa {:?}, {} threads)", t0.elapsed(), isa(), rayon::current_num_threads());
    let base = model.tokenize(&"The agent called read_file and then retried the same call. ".repeat(200))?;
    // Texts are slices of `base` at offsets 0..7.
    let check_len = |n: usize| -> Result<()> {
        ensure!(n + 7 <= base.len(), "{n} tokens is longer than the {}-token test text allows", base.len() - 7);
        Ok(())
    };
    match cmd {
        Cmd::Embed(texts) => {
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            for out in model.embed_batch(&texts)? {
                println!("{}", serde_json::to_string(&out.embedding)?);
            }
        }
        Cmd::Bench(lens) => {
            for &n in &lens {
                let ids = &base[..n.min(base.len())];
                let n = ids.len();
                model.forward(ids); // warm-up
                let iters = (2048 / n).clamp(3, 50);
                let t = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(model.forward(ids));
                }
                let dt = t.elapsed().as_secs_f64() / iters as f64;
                println!("{n:>5} tokens: {:>8.2} ms/text  {:>8.0} tokens/s", dt * 1e3, n as f64 / dt);
                // EG2_TRACE=1: where a pass's time goes, per op (summed over the 24 layers), averaged over the iterations.
                let mut ops = eg2cpu::take_trace();
                if !ops.is_empty() {
                    ops.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let per = (iters + 1) as f64;
                    let total: f64 = ops.iter().map(|o| o.1).sum::<f64>() / per;
                    for (name, secs, calls) in ops {
                        println!(
                            "        {name:<16} {:>7.2} ms  {:>4.0}%  ({} calls/pass)",
                            secs / per * 1e3,
                            secs / per / total * 100.0,
                            calls as f64 / per
                        );
                    }
                    println!(
                        "        {:<16} {:>7.2} ms in traced ops, {:.2} ms elsewhere",
                        "",
                        total * 1e3,
                        (dt - total) * 1e3
                    );
                }
            }
            // Throughput when many texts are embedded at once (texts run concurrently).
            for &n in &[32usize, 128] {
                let batch: Vec<&[u32]> = (0..256).map(|i| &base[i % 7..i % 7 + n]).collect();
                let t = Instant::now();
                let outs = model.forward_batch(&batch);
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(outs);
                println!("batch 256 x {n:>4}: {:>8.1} texts/s  {:>8.0} tokens/s", 256.0 / dt, (256 * n) as f64 / dt);
            }
        }
        Cmd::Image { path, budgets } => {
            let bytes = std::fs::read(&path).with_context(|| path.clone())?;
            let t = Instant::now();
            model.embed_image_with(&bytes, budgets[0])?;
            println!("first image (loads the vision tower): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
            eg2cpu::take_trace();
            for &n in &budgets {
                model.embed_image_with(&bytes, n)?; // warm-up
                eg2cpu::take_trace();
                let iters = (2800 / n).clamp(5, 40);
                let (mut pre, mut vis, mut txt, mut all) = (vec![], vec![], vec![], vec![]);
                let (c0, w0) = (cpu_secs(), Instant::now());
                for _ in 0..iters {
                    let t = Instant::now();
                    let img = model.preprocess_image(&bytes, n)?;
                    let t1 = Instant::now();
                    let soft = model.image_soft_tokens(&img)?;
                    let t2 = Instant::now();
                    std::hint::black_box(model.embed_soft_tokens(&soft));
                    let t3 = Instant::now();
                    pre.push((t1 - t).as_secs_f64() * 1e3);
                    vis.push((t2 - t1).as_secs_f64() * 1e3);
                    txt.push((t3 - t2).as_secs_f64() * 1e3);
                    all.push((t3 - t).as_secs_f64() * 1e3);
                }
                let cores = (cpu_secs() - c0) / w0.elapsed().as_secs_f64();
                let img = model.preprocess_image(&bytes, n)?;
                for v in [&mut pre, &mut vis, &mut txt, &mut all] {
                    v.sort_by(f64::total_cmp);
                }
                println!(
                    "{n:>4} soft tokens ({}x{} px, {} patches): {:>6.1} ms/image (p50; min {:.1})  = preprocess {:.1} + vision {:.1} + text {:.1}  ({cores:.1} cores busy)",
                    img.height,
                    img.width,
                    img.height * img.width / 256,
                    quantile(&all, 0.5),
                    all[0],
                    quantile(&pre, 0.5),
                    quantile(&vis, 0.5),
                    quantile(&txt, 0.5),
                );
                // EG2_TRACE=1: per op, averaged over the iterations.
                let mut ops = eg2cpu::take_trace();
                if !ops.is_empty() {
                    ops.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let per = iters as f64;
                    for (name, secs, calls) in ops {
                        println!(
                            "        {name:<22} {:>7.2} ms  ({} calls/image)",
                            secs / per * 1e3,
                            calls as f64 / per
                        );
                    }
                }
            }
        }
        Cmd::Audio { path, seconds } => {
            let clip = eg2cpu::decode_wav(&std::fs::read(&path).with_context(|| path.clone())?)?;
            ensure!(!clip.is_empty(), "{path}: no samples");
            let t = Instant::now();
            model.embed_audio_samples(&clip)?;
            println!("first clip (loads the audio tower): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
            eg2cpu::take_trace();
            for &secs in &seconds {
                let n = secs * eg2cpu::SAMPLE_RATE as usize;
                let samples: Vec<f32> = clip.iter().copied().cycle().take(n).collect();
                model.embed_audio_samples(&samples)?; // warm-up
                eg2cpu::take_trace();
                let iters = (60 / secs).clamp(3, 20);
                let (mut fe, mut aud, mut txt, mut all) = (vec![], vec![], vec![], vec![]);
                let (c0, w0) = (cpu_secs(), Instant::now());
                for _ in 0..iters {
                    let t = Instant::now();
                    let mel = eg2cpu::log_mel(&samples);
                    let t1 = Instant::now();
                    let soft = model.audio_soft_tokens(&mel)?;
                    let t2 = Instant::now();
                    std::hint::black_box(model.embed_audio_soft_tokens(&soft));
                    let t3 = Instant::now();
                    fe.push((t1 - t).as_secs_f64() * 1e3);
                    aud.push((t2 - t1).as_secs_f64() * 1e3);
                    txt.push((t3 - t2).as_secs_f64() * 1e3);
                    all.push((t3 - t).as_secs_f64() * 1e3);
                }
                let cores = (cpu_secs() - c0) / w0.elapsed().as_secs_f64();
                for v in [&mut fe, &mut aud, &mut txt, &mut all] {
                    v.sort_by(f64::total_cmp);
                }
                println!(
                    "{secs:>3} s ({} soft tokens): {:>7.1} ms/clip (p50; min {:.1})  = features {:.1} + audio {:.1} + text {:.1}  ({cores:.1} cores busy)",
                    eg2cpu::soft_tokens_for(n),
                    quantile(&all, 0.5),
                    all[0],
                    quantile(&fe, 0.5),
                    quantile(&aud, 0.5),
                    quantile(&txt, 0.5),
                );
                // EG2_TRACE=1: per op, averaged over the iterations.
                let mut ops = eg2cpu::take_trace();
                if !ops.is_empty() {
                    ops.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let per = iters as f64;
                    for (name, secs, calls) in ops {
                        println!(
                            "        {name:<22} {:>7.2} ms  ({} calls/clip)",
                            secs / per * 1e3,
                            calls as f64 / per
                        );
                    }
                }
            }
        }
        Cmd::AudioEmbed(paths) => {
            for path in &paths {
                let bytes = std::fs::read(path).with_context(|| path.clone())?;
                println!("{}", serde_json::to_string(&model.embed_audio(&bytes)?.embedding)?);
            }
        }
        Cmd::ImageEmbed(paths) => {
            for path in &paths {
                let bytes = std::fs::read(path).with_context(|| path.clone())?;
                println!("{}", serde_json::to_string(&model.embed_image(&bytes)?.embedding)?);
            }
        }
        Cmd::Batch { texts, len, in_flight } => {
            check_len(len)?;
            let ids: Vec<&[u32]> = (0..texts).map(|i| &base[i % 7..i % 7 + len]).collect();
            model.forward_batch_with(&ids[..ids.len().min(8)], 8); // warm-up
            for k in in_flight {
                let (t, r0) = (Instant::now(), cpu_secs());
                std::hint::black_box(model.forward_batch_with(&ids, k));
                let (dt, cpu) = (t.elapsed().as_secs_f64(), cpu_secs() - r0);
                println!(
                    "batch {texts:>4} x {len:>4} tok  in-flight {k:>3}: {:>7.0} tok/s  {:>6.1} texts/s  cores busy {:>5.1}",
                    (texts * len) as f64 / dt,
                    texts as f64 / dt,
                    cpu / dt
                );
            }
        }
        Cmd::Callers { threads, per, len } => {
            check_len(len)?;
            model.forward(&base[..len]); // warm-up
            let (t, r0) = (Instant::now(), cpu_secs());
            let lat = callers(&model, &base, threads, per, len);
            let (dt, cpu) = (t.elapsed().as_secs_f64(), cpu_secs() - r0);
            println!(
                "{threads:>3} callers x {len:>4} tok: {:>7.0} tok/s  {:>6.1} req/s  latency p50 {:>7.1} ms  p95 {:>7.1} ms  cores busy {:>5.1}",
                (lat.len() * len) as f64 / dt,
                lat.len() as f64 / dt,
                quantile(&lat, 0.5),
                quantile(&lat, 0.95),
                cpu / dt
            );
        }
        Cmd::Exit { upto, lens } => {
            let nl = model.cfg.layers.len();
            ensure!(upto < nl, "layer {upto} is out of range: the model has layers 0..{nl}");
            for &n in &lens {
                check_len(n)?;
            }
            for &n in &lens {
                let one: Vec<&[u32]> = vec![&base[..n]];
                model.forward_batch_partial(&one, upto); // warm-up
                let iters = (2048 / n).clamp(5, 50);
                let t = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(model.forward_batch_partial(&one, upto));
                }
                let single = t.elapsed().as_secs_f64() / iters as f64;
                let batch: Vec<&[u32]> = (0..256).map(|i| &base[i % 7..i % 7 + n]).collect();
                model.forward_batch_partial(&batch, upto); // warm-up
                let t = Instant::now();
                std::hint::black_box(model.forward_batch_partial(&batch, upto));
                let part = t.elapsed().as_secs_f64() / 256.0;
                let t = Instant::now();
                std::hint::black_box(model.forward_batch(&batch));
                let full = t.elapsed().as_secs_f64() / 256.0;
                println!(
                    "exit after layer {upto}, {n:>4} tokens: 1 text {:>7.2} ms | 256 texts {:>6.2} ms/text ({:>7.0} tokens/s) | full pass {:>6.2} ms/text",
                    single * 1e3,
                    part * 1e3,
                    n as f64 / part,
                    full * 1e3
                );
            }
        }
        Cmd::Mixed { threads, per, len } => {
            check_len(len)?;
            let big: Vec<&[u32]> = (0..512).map(|i| &base[i % 7..i % 7 + 128]).collect();
            let t = Instant::now();
            let (lat, bt) = std::thread::scope(|s| {
                let b = s.spawn(|| {
                    let t = Instant::now();
                    std::hint::black_box(model.forward_batch(&big));
                    t.elapsed().as_secs_f64()
                });
                std::thread::sleep(std::time::Duration::from_millis(200)); // let the batch get going
                let lat = callers(&model, &base, threads, per, len);
                (lat, b.join().unwrap())
            });
            println!(
                "{threads:>3} callers x {len:>4} tok during a 512 x 128 batch: single p50 {:>7.1} ms  p95 {:>7.1} ms; batch took {bt:.1} s (total {:.1} s)",
                quantile(&lat, 0.5),
                quantile(&lat, 0.95),
                t.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
