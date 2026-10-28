// EmbeddingGemma 2 image embeddings on WebGPU: weights from weights.bin/manifest.json (see convert.py), the forward
// pass as a list of compute dispatches recorded once per image size and replayed in one command buffer per image.
import * as K from './kernels.js';

const PATCH = 16, POOL = 3;

export async function load(base = '.', { subgroupMatrix = true, onProgress = null } = {}) {
  const t0 = performance.now();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
    },
    requiredFeatures: ['timestamp-query', 'subgroups', 'shader-f16', 'chromium-experimental-subgroup-matrix'].filter((f) => adapter.features.has(f)),
  });
  device.lost.then((i) => console.error('device lost', i.message));
  const man = await (await fetch(`${base}/manifest.json`)).json();
  const tFetch0 = performance.now();
  const bin = await fetchBytes(`${base}/weights.bin`, onProgress);
  const tFetch = performance.now() - tFetch0;
  const W = {};
  for (const [name, e] of Object.entries(man.tensors)) {
    const buf = device.createBuffer({ size: Math.ceil(e.nbytes / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buf, 0, bin, e.offset, Math.ceil(e.nbytes / 4) * 4);
    W[name] = buf;
  }
  const special = new Float32Array(bin, man.tensors['t.special'].offset, 4 * 512);
  const m = new Model(device, adapter, man, W, special.slice());
  m.useSG = subgroupMatrix && device.features.has('chromium-experimental-subgroup-matrix');
  await device.queue.onSubmittedWorkDone();
  m.loadMs = { total: performance.now() - t0, fetch: tFetch };
  return m;
}

const f32bits = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];

// cos/sin tables [t, dim/2] as eg2cpu's Rope: inv_freq = theta^(-2i/dim) (exponent in f32), angle pos*f in f32.
// The whole file as one ArrayBuffer, telling onProgress(bytes so far, total) as it arrives.
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!onProgress || !total || !res.body) return res.arrayBuffer();
  const buf = new Uint8Array(total);
  let at = 0;
  for (const reader = res.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, at);
    at += value.length;
    onProgress(at, total);
  }
  return buf.buffer;
}

function ropeTables(theta, dim, t) {
  const half = dim / 2;
  const inv = Array.from({ length: half }, (_, i) => Math.fround(1 / Math.pow(theta, Math.fround((2 * i) / dim))));
  const cos = new Float32Array(t * half), sin = new Float32Array(t * half);
  for (let p = 0; p < t; p++) for (let i = 0; i < half; i++) {
    const a = Math.fround(p * inv[i]);
    cos[p * half + i] = Math.cos(a);
    sin[p * half + i] = Math.sin(a);
  }
  return { cos, sin };
}

class Model {
  constructor(device, adapter, man, W, special) {
    this.device = device;
    this.adapter = adapter;
    this.man = man;
    this.W = W;
    this.vc = man.config.vision_config;
    this.tc = man.config.text_config;
    this.special = special;
    this.pipes = new Map();
    this.plans = new Map();
  }

  pipe(key, code) {
    let p = this.pipes.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipes.set(key, p);
    }
    return p;
  }

  buf(floats, usage = 0) {
    return this.device.createBuffer({ size: Math.max(16, floats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | usage });
  }

  data(arr) {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  // One dispatch: pipeline, uniform values, storage buffers (bindings 1..), workgroup counts.
  op(plan, key, code, u, bufs, groups, label) {
    const pipeline = this.pipe(key, code);
    const ub = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(ub, 0, new Uint32Array(u));
    const entries = [{ binding: 0, resource: { buffer: ub } }, ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))];
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    plan.steps.push({ pipeline, bg, groups, label, key });
  }

  // C[M, N] (+)= A[M, K] W[N, K]^T
  // C[M, N] (+)= A[M, K] W[N, K]^T. Tall M (the vision tower): 64 x 64 tiles, 8 x 4 per thread. Short M (the text
  // tower): one tile row covering all of M, split along K into enough workgroups to fill the GPU, then summed.
  mmW(plan, A, Wt, C, M, N, Kd, { lda = Kd, ldc = N, aOff = 0, cOff = 0, add = false, label, padded = false } = {}) {
    if (Kd % 16 || N % 64 || lda % 4 || aOff % 4 || ldc % 4 || cOff % 4) throw new Error(`mmW shape ${M}x${N}x${Kd}`);
    label = label || `${N}x${Kd}`;
    let cfg, split = 1;
    if (M > 128 && padded && this.useSG && !add && N % 32 === 0) {
      // subgroup-matrix tiles store whole 32-row blocks: C must have rows padded to 32 (plan buffers do)
      const sg = { BM: 32, BN: 32, BK: 16 };
      this.op(plan, `mmSG_${JSON.stringify(sg)}`, K.matmulSG(sg), [M, N, Kd, lda, ldc, aOff, cOff, Kd, 0], [A, Wt, C],
        [N / sg.BN, Math.ceil(M / sg.BM), 1], label);
      return;
    }
    if (M > 128) cfg = { BM: 64, BN: 64, TM: 8, TN: 4 };
    else {
      const BM = Math.ceil(M / 16) * 16;
      cfg = { BM, BN: 64, TM: BM / 16, TN: 4 };
      const tiles = N / 64;
      while (split < 8 && tiles * split < 128 && (Kd / (split * 2)) % 16 === 0 && Kd / (split * 2) >= 64) split *= 2;
    }
    const groups = [N / cfg.BN, Math.ceil(M / cfg.BM), split];
    const kslice = Kd / split;
    const key = `mmW2_${JSON.stringify(cfg)}_${add && split === 1 ? 1 : 0}`;
    const code = K.matmulW2({ ...cfg, mode: add && split === 1 ? 1 : 0 });
    if (split === 1) {
      this.op(plan, key, code, [M, N, Kd, lda, ldc, aOff, cOff, kslice, 0], [A, Wt, C], groups, label);
      return;
    }
    if (split * M * N > plan.partialFloats) throw new Error('partial buffer too small');
    this.op(plan, key, code, [M, N, Kd, lda, N, aOff, 0, kslice, M * N], [A, Wt, plan.B.partial], groups, label);
    const nv = (M * N) / 4;
    this.op(plan, `reduceK${add ? 1 : 0}`, K.reduceK(add ? 1 : 0), [M, N, split, ldc, cOff], [plan.B.partial, C],
      [Math.min(Math.ceil(nv / 256), 65535), Math.ceil(nv / 256 / 65535), 1], label + ':reduce');
  }

  mmF(plan, transB, A, B, C, u, groups) {
    this.op(plan, `mmF${transB}`, K.matmulF32(transB), u, [A, B, C], groups);
  }

  // Multi-head attention from a [t, (nq + 2 nkv) hd] qkv buffer into out [t, nq hd], chunked over heads so S stays
  // under ~64 MB.
  attention(plan, qkv, out, S, t, nq, nkv, hd, window) {
    const row = (nq + 2 * nkv) * hd;
    const per = Math.max(1, Math.min(nq, Math.floor(this.sFloats / (t * t))));
    const grp = nq / nkv;
    for (let h0 = 0; h0 < nq; h0 += per) {
      const nh = Math.min(per, nq - h0);
      // S[h] = Q_h K_g^T. Head h's K is at column (nq + h/grp)*hd; with batch b = h - h0 the kv index is
      // (h0 + b)/grp, so the B offset is folded per chunk when grp divides h0.
      const g0 = Math.floor(h0 / grp);
      if (h0 % grp && per > 1) throw new Error('head chunk not aligned to KV groups');
      this.mmF(plan, true, qkv, qkv, S,
        [t, t, hd, row, row, t, h0 * hd, (nq + g0) * hd, 0, hd, hd, t * t, grp],
        [Math.ceil(t / 64), Math.ceil(t / 64), nh]);
      const rows = nh * t;
      this.op(plan, 'softmax', K.softmax, [t, t, t, window], [S], [Math.min(rows, 65535), Math.ceil(rows / 65535), 1]);
      this.mmF(plan, false, S, qkv, out,
        [t, hd, t, t, row, nq * hd, 0, (nq + nkv + g0) * hd, h0 * hd, t * t, hd, hd, grp],
        [Math.ceil(hd / 64), Math.ceil(t / 64), nh]);
    }
  }

  // The dispatch list for one image size, with its activation buffers.
  plan(height, width) {
    const key = `${height}x${width}`;
    if (this.plans.has(key)) return this.plans.get(key);
    const { vc, tc, W } = this;
    const gh = height / PATCH, gw = width / PATCH, t = gh * gw;
    const d = vc.hidden_size, nh = vc.num_attention_heads, hd = vc.head_dim, inter = vc.intermediate_size;
    const n = (gh / POOL) * (gw / POOL), td = tc.hidden_size, tt = n + 4, pd = tc.hidden_size_per_layer_input;
    const nl = tc.num_hidden_layers;
    if (gh % POOL || gw % POOL || Math.max(gh, gw) > this.man.pos_rows) throw new Error(`bad image size ${key}`);
    this.sFloats = 16 * 1024 * 1024;
    const plan = { key, steps: [], marks: [], gh, gw, t, n, tt };
    const mark = (label, buffer, rows, cols) => plan.marks.push({ label, step: plan.steps.length, buffer, rows, cols });
    const tp = Math.ceil(t / 32) * 32;  // vision rows padded for whole-block matrix stores
    const B = {
      img: this.buf(Math.ceil(height * width * 3 / 4)),
      patches: this.buf(tp * 768), h: this.buf(tp * d), x: this.buf(tp * d), qkv: this.buf(tp * 3 * d), attn: this.buf(tp * d),
      o: this.buf(tp * d), gu: this.buf(tp * 2 * inter), g: this.buf(tp * inter),
      S: this.buf(this.sFloats),
      pooled: this.buf(n * d),
      partial: this.buf(8 * (tt + 16) * 4096),
      th: this.buf(tt * td), tin: this.buf(tt * td), tx: this.buf(tt * td), tqkv: this.buf(tt * 3072), tattn: this.buf(tt * 2048),
      to: this.buf(tt * td), tgu: this.buf(tt * 4096), tg: this.buf(tt * 2048), ple: this.buf(tt * nl * pd), pleRaw: this.buf(tt * nl * pd), pg: this.buf(tt * pd),
      out: this.buf(tc.embedding_dim),
    };
    plan.B = B;
    plan.partialFloats = 8 * (tt + 16) * 4096;
    plan.readback = this.device.createBuffer({ size: tc.embedding_dim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const vr = ropeTables(vc.rope_parameters.rope_theta, hd / 2, Math.max(gh, gw));
    const vcos = this.data(vr.cos), vsin = this.data(vr.sin);

    // ---- vision tower
    this.op(plan, 'patchify', K.patchify, [gw, width, t], [B.img, B.patches], [3, t, 1]);
    mark('patches', B.patches, t, 768);
    this.mmW(plan, B.patches, W['v.input_proj'], B.h, t, d, 768, { padded: true });
    mark('v_inproj', B.h, t, d);
    this.op(plan, 'addPos', K.addPos, [t, d, gw, this.man.pos_rows], [B.h, W['v.pos']], [Math.ceil(d / 256), t, 1]);
    mark('v_embed', B.h, t, d);
    const rowsGrid = (r) => [Math.min(r, 65535), Math.ceil(r / 65535), 1];
    this.op(plan, 'norm', K.norm('norm'), [d, d, d, 0, 0, f32bits(1)], [W['v0.input_layernorm'], B.h, B.x], rowsGrid(t));
    for (let L = 0; L < vc.num_hidden_layers; L++) {
      const P = `v${L}.`;
      this.mmW(plan, B.x, W[P + 'wqkv'], B.qkv, t, 3 * d, d, { padded: true });
      if (hd !== 64) throw new Error('vision kernels assume head_dim 64');
      this.op(plan, 'qkvAxial64', K.qkvAxial64, [nh, gw, 3 * nh],
        [B.qkv, W[P + 'self_attn.q_norm'], W[P + 'self_attn.k_norm'], vcos, vsin], [(t * 3 * nh) / 4, 1, 1]);
      this.op(plan, 'flash64', K.flash64, [t, nh], [B.qkv, B.attn], [Math.ceil(t / 32), nh, 1]);
      this.mmW(plan, B.attn, W[P + 'wo'], B.o, t, d, d, { padded: true });
      this.op(plan, 'addnorm', K.norm('addnorm'), [d], [W[P + 'post_attention_layernorm'], B.o, B.h, B.x, W[P + 'pre_feedforward_layernorm']], rowsGrid(t));
      this.mmW(plan, B.x, W[P + 'wgu'], B.gu, t, 2 * inter, d, { padded: true });
      this.op(plan, 'gelu', K.geluMul, [t, inter, 2 * inter, 2 * inter, inter, 0, inter], [B.gu, B.gu, B.g], [Math.ceil(inter / 256), t, 1]);
      this.mmW(plan, B.g, W[P + 'wd'], B.o, t, d, inter, { padded: true });
      // h += norm(o); x = next layer's input norm of h
      const next = L + 1 < vc.num_hidden_layers ? W[`v${L + 1}.input_layernorm`] : W[P + 'input_layernorm'];
      this.op(plan, 'addnorm', K.norm('addnorm'), [d], [W[P + 'post_feedforward_layernorm'], B.o, B.h, B.x, next], rowsGrid(t));
      mark(`v_layer${L}`, B.h, t, d);
    }
    this.op(plan, 'pool', K.pool, [d, gw, gw / POOL, POOL], [B.h, B.pooled], [n, 1, 1]);
    // soft tokens straight into rows 2.. of the text tower's input
    this.mmW(plan, B.pooled, W['v.proj'], B.th, n, td, d, { cOff: 2 * td });
    plan.textStart = plan.steps.length;
    mark('soft', B.th, tt, td);  // read rows 2..n+2

    // ---- text tower over <bos> <boi> soft <eoi> <eos>
    const sp = new Float32Array(4 * td);
    const es = Math.fround(Math.sqrt(td));
    for (let i = 0; i < sp.length; i++) sp[i] = Math.fround(this.special[i] * es);
    plan.special = this.data(sp);
    // (the copies of the special rows and of the inputs happen in encode(), at textStart)
    this.mmW(plan, B.tin, W['t.ple_model_proj'], B.pleRaw, tt, nl * pd, td);
    this.op(plan, 'norm', K.norm('norm'), [pd, pd, pd, 0, 0, f32bits(Math.fround(Math.pow(td, -0.5)))],
      [W['t.ple_proj_norm'], B.pleRaw, B.ple], rowsGrid(tt * nl));
    const tcos = {}, tsin = {};
    for (const [kind, dim] of [['sliding_attention', tc.head_dim], ['full_attention', 512]]) {
      const r = ropeTables(tc.rope_parameters[kind].rope_theta, dim, tt);
      tcos[kind] = this.data(r.cos);
      tsin[kind] = this.data(r.sin);
    }
    this.op(plan, 'norm', K.norm('norm'), [td, td, td, 0, 0, f32bits(1)], [W['t0.input_layernorm'], B.th, B.tx], rowsGrid(tt));
    const tq = tc.num_attention_heads;
    for (let L = 0; L < nl; L++) {
      const P = `t${L}.`;
      const kind = tc.layer_types[L];
      const o = tc.per_layer_config[String(L).padStart(2, '0')] || {};
      const lhd = o.head_dim || tc.head_dim, kvh = o.num_key_value_heads || tc.num_key_value_heads;
      const qd = tq * lhd, row = (tq + 2 * kvh) * lhd;
      this.mmW(plan, B.tx, W[P + 'wqkv'], B.tqkv, tt, row, td);
      this.op(plan, 'qkv1d', K.qkvPrep(false), [lhd, tq, kvh, 1],
        [B.tqkv, W[P + 'self_attn.q_norm'], W[P + 'self_attn.k_norm'], tcos[kind], tsin[kind]], [tt, tq + 2 * kvh, 1]);
      const window = kind === 'full_attention' ? 1 << 30 : tc.sliding_window;
      this.attention(plan, B.tqkv, B.tattn, B.S, tt, tq, kvh, lhd, window);
      this.mmW(plan, B.tattn, W[P + 'wo'], B.to, tt, td, qd);
      this.op(plan, 'addnorm', K.norm('addnorm'), [td], [W[P + 'post_attention_layernorm'], B.to, B.th, B.tx, W[P + 'pre_feedforward_layernorm']], rowsGrid(tt));
      const ti = tc.intermediate_size;
      this.mmW(plan, B.tx, W[P + 'wgu'], B.tgu, tt, 2 * ti, td);
      this.op(plan, 'gelu', K.geluMul, [tt, ti, 2 * ti, 2 * ti, ti, 0, ti], [B.tgu, B.tgu, B.tg], [Math.ceil(ti / 256), tt, 1]);
      this.mmW(plan, B.tg, W[P + 'wd'], B.to, tt, td, ti);
      // h += norm(o); tx = scratch (overwritten below)
      this.op(plan, 'addnorm', K.norm('addnorm'), [td], [W[P + 'post_feedforward_layernorm'], B.to, B.th, B.tx, W[P + 'pre_feedforward_layernorm']], rowsGrid(tt));
      // per-layer-embedding block
      this.mmW(plan, B.th, W[P + 'ple_gate'], B.pg, tt, pd, td);
      this.op(plan, 'gelu', K.geluMul, [tt, pd, pd, nl * pd, pd, 0, L * pd], [B.pg, B.ple, B.tg], [Math.ceil(pd / 256), tt, 1]);
      this.mmW(plan, B.tg, W[P + 'ple_proj'], B.to, tt, td, pd);
      this.op(plan, 'addscale', K.norm('addscale'), [td], [W[P + 'ple_block.post_per_layer_input_norm'], B.to, B.th, W[P + 'layer_scalar']], rowsGrid(tt));
      mark(`t_layer${L}`, B.th, tt, td);
      const next = L + 1 < nl ? W[`t${L + 1}.input_layernorm`] : W['t.norm'];
      this.op(plan, 'norm', K.norm('norm'), [td, td, td, 0, 0, f32bits(1)], [next, B.th, B.tx], rowsGrid(tt));
    }
    this.op(plan, 'meanProj', K.meanProj, [tt, td, tc.embedding_dim], [B.tx, W['t.out_proj'], B.out], [1, 1, 1]);
    mark('embedding', B.out, 1, tc.embedding_dim);
    this.plans.set(key, plan);
    return plan;
  }

  // Encode steps [0, upto) into one command buffer (one compute pass per dispatch run).
  encode(plan, upto = plan.steps.length, timestamps = null) {
    const enc = this.device.createCommandEncoder();
    const td = this.tc.hidden_size;
    let pass = null;
    const flush = () => { if (pass) { pass.end(); pass = null; } };
    for (let i = 0; i < upto; i++) {
      if (i === plan.textStart) {
        flush();
        enc.copyBufferToBuffer(plan.special, 0, plan.B.th, 0, 2 * td * 4);
        enc.copyBufferToBuffer(plan.special, 2 * td * 4, plan.B.th, (plan.n + 2) * td * 4, 2 * td * 4);
        enc.copyBufferToBuffer(plan.B.th, 0, plan.B.tin, 0, plan.tt * td * 4);
      }
      if (!pass) pass = enc.beginComputePass();
      const s = plan.steps[i];
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(...s.groups);
    }
    flush();
    return enc;
  }

  // GPU time per kernel kind (timestamp queries around each dispatch), summed over steps; ms.
  async profile(rgb, height, width) {
    const plan = this.plan(height, width);
    this.setImage(plan, rgb);
    const n = plan.steps.length;
    const qs = this.device.createQuerySet({ type: 'timestamp', count: 2 * n });
    const res = this.device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = this.device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    const td = this.tc.hidden_size;
    for (let i = 0; i < n; i++) {
      if (i === plan.textStart) {
        enc.copyBufferToBuffer(plan.special, 0, plan.B.th, 0, 2 * td * 4);
        enc.copyBufferToBuffer(plan.special, 2 * td * 4, plan.B.th, (plan.n + 2) * td * 4, 2 * td * 4);
        enc.copyBufferToBuffer(plan.B.th, 0, plan.B.tin, 0, plan.tt * td * 4);
      }
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      const s = plan.steps[i];
      pass.setPipeline(s.pipeline); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.groups);
      pass.end();
    }
    enc.resolveQuerySet(qs, 0, 2 * n, res, 0);
    enc.copyBufferToBuffer(res, 0, rb, 0, 16 * n);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const ts = new BigInt64Array(rb.getMappedRange().slice(0));
    rb.unmap();
    const by = {};
    let total = 0;
    for (let i = 0; i < n; i++) {
      const k = (i < plan.textStart ? 'v:' : 't:') + plan.steps[i].key + (plan.steps[i].label ? ':' + plan.steps[i].label : '');
      const ms = Number(ts[2 * i + 1] - ts[2 * i]) / 1e6;
      by[k] = (by[k] || 0) + ms;
      total += ms;
    }
    const span = Number(ts[2 * n - 1] - ts[0]) / 1e6;
    return { total, span, steps: n, by };
  }

  setImage(plan, rgb) {
    const padded = new Uint8Array(Math.ceil(rgb.length / 4) * 4);
    padded.set(rgb);
    this.device.queue.writeBuffer(plan.B.img, 0, padded);
  }

  // Embed RGB bytes already at a valid target size. Returns Float32Array(768).
  async embedRgb(rgb, height, width) {
    const plan = this.plan(height, width);
    this.setImage(plan, rgb);
    const enc = this.encode(plan);
    enc.copyBufferToBuffer(plan.B.out, 0, plan.readback, 0, plan.readback.size);
    this.device.queue.submit([enc.finish()]);
    await plan.readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(plan.readback.getMappedRange().slice(0));
    plan.readback.unmap();
    return out;
  }

  // Run up to a mark and read its buffer (for layer-by-layer comparison).
  async probe(rgb, height, width, label) {
    const plan = this.plan(height, width);
    const mk = plan.marks.find((m) => m.label === label);
    if (!mk) throw new Error(`no mark ${label}`);
    this.setImage(plan, rgb);
    const enc = this.encode(plan, mk.step);
    const size = mk.rows * mk.cols * 4;
    const rb = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(mk.buffer, 0, rb, 0, size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return out;
  }
}

// ---- preprocessing (port of eg2cpu vision.rs: target_size + antialiased bicubic, bit-exact on the same RGB)

export function targetSize(h, w, maxPatches) {
  const targetPx = maxPatches * PATCH * PATCH;
  const factor = Math.sqrt(targetPx / (h * w));
  const side = POOL * PATCH;
  let th = Math.floor((factor * h) / side) * side, tw = Math.floor((factor * w) / side) * side;
  const maxSide = Math.floor(maxPatches / (POOL * POOL)) * side;
  if (th === 0 && tw === 0) throw new Error('image too small');
  if (th === 0) { th = side; tw = Math.min(Math.floor(w / h) * side, maxSide); }
  else if (tw === 0) { tw = side; th = Math.min(Math.floor(h / w) * side, maxSide); }
  return [th, tw];
}

function cubic(x) {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function resizeWeights(insz, outsz) {
  const scale = insz / outsz;
  const [support, invscale] = scale >= 1 ? [2 * scale, 1 / scale] : [2, 1];
  const maxInterp = Math.ceil(support) * 2 + 1;
  let wtMax = 0;
  const rows = [];
  for (let i = 0; i < outsz; i++) {
    const center = scale * (i + 0.5);
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xsize = Math.min(Math.max(Math.min(Math.trunc(center + support + 0.5), insz) - xmin, 0), maxInterp);
    let w = Array.from({ length: xsize }, (_, j) => cubic((j + xmin - center + 0.5) * invscale));
    const tot = w.reduce((a, b) => a + b, 0);
    if (tot !== 0) w = w.map((v) => v / tot);
    for (const v of w) wtMax = Math.max(wtMax, v);
    rows.push([xmin, w]);
  }
  let prec = 0;
  while (prec < 22 && Math.trunc(0.5 + wtMax * 2 ** (prec + 1)) < 1 << 15) prec++;
  return [rows.map(([xmin, w]) => [xmin, Int32Array.from(w, (v) => { v *= 2 ** prec; return v < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v); })]), prec];
}

// Resize along the middle axis of [outer, n, inner] uint8.
function resizeAxis(src, outer, n, inner, out) {
  const [ws, prec] = resizeWeights(n, out);
  const round = 1 << (prec - 1);
  const dst = new Uint8Array(outer * out * inner);
  const acc = new Int32Array(inner);
  for (let o = 0; o < outer; o++) for (let i = 0; i < out; i++) {
    const [xmin, w] = ws[i];
    acc.fill(round);
    for (let j = 0; j < w.length; j++) {
      const s = (o * n + xmin + j) * inner, wj = w[j];
      for (let c = 0; c < inner; c++) acc[c] += src[s + c] * wj;
    }
    const dd = (o * out + i) * inner;
    for (let c = 0; c < inner; c++) dst[dd + c] = Math.min(255, Math.max(0, acc[c] >> prec));
  }
  return dst;
}

export function resize(rgb, h, w, th, tw) {
  let data = rgb;
  if (tw !== w) data = resizeAxis(data, h, w, 3, tw);
  if (th !== h) data = resizeAxis(data, 1, h, tw * 3, th);
  return data;
}

// Decode any browser-supported image to RGB bytes, resize to the budget's target size.
export async function preprocess(blob, maxSoftTokens = 70) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const rgb = new Uint8Array(bmp.width * bmp.height * 3);
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) { rgb[j] = px[i]; rgb[j + 1] = px[i + 1]; rgb[j + 2] = px[i + 2]; }
  const [th, tw] = targetSize(bmp.height, bmp.width, maxSoftTokens * POOL * POOL);
  return { rgb: resize(rgb, bmp.height, bmp.width, th, tw), height: th, width: tw };
}
