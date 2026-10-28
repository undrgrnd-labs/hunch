// WGSL compute kernels for the EmbeddingGemma 2 image path. Activations are f32; weight matrices are bf16 stored as
// u32 pairs and widened in the shader (bits << 16), so no shader-f16 is needed. Every kernel takes its sizes from a
// uniform block of u32 fields named in the struct.

const uniforms = (fields) => `struct P { ${fields.map((f) => `${f}: u32`).join(', ')} }
@group(0) @binding(0) var<uniform> p: P;`;

// C[m, n] = sum_k A[m, k] * W[n, k]: f32 activations times a bf16 [N, K] weight. Tiles of BM x 64 (BM = 64 or 32),
// 16 deep; 256 threads, each with (BM/16) x 4 accumulators. Needs K % 16 == 0, N % 64 == 0, lda % 4 == 0.
// MODE 0: C = AW; MODE 1: C += AW (residual add).
export const matmulW = (BM, mode = 0) => `${uniforms(['M', 'N', 'K', 'lda', 'ldc', 'aOff', 'cOff'])}
@group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
const BM = ${BM}u;
const RM = ${BM / 16}u;
var<workgroup> As: array<f32, ${BM * 16}>;  // [k][m]
var<workgroup> Bs: array<f32, ${64 * 16}>;  // [k][n]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * BM;
  let n0 = wg.x * 64u;
  let tx = li % 16u;  // column group: n = n0 + tx*4 .. +4
  let ty = li / 16u;  // row group: m = m0 + ty*RM .. +RM
  var acc: array<vec4<f32>, ${BM / 16}>;
  for (var i = 0u; i < RM; i++) { acc[i] = vec4<f32>(0.0); }
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    // A tile: BM rows x 16 k = BM*4 vec4s, one per thread (BM=64) or every other thread (BM=32).
    if (li < BM * 4u) {
      let r = li / 4u;
      let kq = (li % 4u) * 4u;
      var av = vec4<f32>(0.0);
      if (m0 + r < p.M) { av = A[(p.aOff + (m0 + r) * p.lda + k0 + kq) / 4u]; }
      // store transposed: As[k][r]
      As[(kq + 0u) * BM + r] = av.x;
      As[(kq + 1u) * BM + r] = av.y;
      As[(kq + 2u) * BM + r] = av.z;
      As[(kq + 3u) * BM + r] = av.w;
    }
    // W tile: 64 rows x 16 k = 256 groups of 4 bf16.
    {
      let r = li / 4u;
      let kq = (li % 4u) * 4u;
      let wv = W[((n0 + r) * p.K + k0 + kq) / 4u];
      Bs[(kq + 0u) * 64u + r] = bitcast<f32>(wv.x << 16u);
      Bs[(kq + 1u) * 64u + r] = bitcast<f32>(wv.x & 0xffff0000u);
      Bs[(kq + 2u) * 64u + r] = bitcast<f32>(wv.y << 16u);
      Bs[(kq + 3u) * 64u + r] = bitcast<f32>(wv.y & 0xffff0000u);
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      let bo = k * 64u + tx * 4u;
      let b = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
      let ao = k * BM + ty * RM;
      for (var i = 0u; i < RM; i++) { acc[i] += As[ao + i] * b; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < RM; i++) {
    let m = m0 + ty * RM + i;
    if (m < p.M) {
      let c = p.cOff + m * p.ldc + n0 + tx * 4u;
      ${mode === 1
        ? 'C[c] += acc[i].x; C[c + 1u] += acc[i].y; C[c + 2u] += acc[i].z; C[c + 3u] += acc[i].w;'
        : 'C[c] = acc[i].x; C[c + 1u] = acc[i].y; C[c + 2u] = acc[i].z; C[c + 3u] = acc[i].w;'}
    }
  }
}`;

// Batched f32 matmul for attention, any sizes: C[b][m, n] = sum_k A[b][m, k] * B[b'][.], b' = b / bDiv.
// TRANSB: B[n, k] at bOff + b'*bBatch + n*ldb + k (Q K^T); else B[k, n] at ... + k*ldb + n (P V).
export const matmulF32 = (transB) => `${uniforms(['M', 'N', 'K', 'lda', 'ldb', 'ldc', 'aOff', 'bOff', 'cOff', 'aBatch', 'bBatch', 'cBatch', 'bDiv'])}
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<array<f32, 64>, 16>;
var<workgroup> Bs: array<array<f32, 64>, 16>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let b = wg.z;
  let m0 = wg.y * 64u;
  let n0 = wg.x * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  let aBase = p.aOff + b * p.aBatch;
  let bBase = p.bOff + (b / p.bDiv) * p.bBatch;
  var acc: array<array<f32, 4>, 4>;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var e = 0u; e < 4u; e++) {
      let idx = li + e * 256u;
      let r = idx / 16u;
      let kk = idx % 16u;
      var v = 0.0;
      if (m0 + r < p.M && k0 + kk < p.K) { v = A[aBase + (m0 + r) * p.lda + k0 + kk]; }
      As[kk][r] = v;
      ${transB
        ? `var w = 0.0;
      if (n0 + r < p.N && k0 + kk < p.K) { w = B[bBase + (n0 + r) * p.ldb + k0 + kk]; }
      Bs[kk][r] = w;`
        : `let kk2 = idx / 64u;
      let n = idx % 64u;
      var w = 0.0;
      if (n0 + n < p.N && k0 + kk2 < p.K) { w = B[bBase + (k0 + kk2) * p.ldb + n0 + n]; }
      Bs[kk2][n] = w;`}
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      var a: array<f32, 4>;
      var bb: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[k][ty + 16u * i]; bb[i] = Bs[k][tx + 16u * i]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i][j] += a[i] * bb[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty + 16u * i;
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tx + 16u * j;
      if (m < p.M && n < p.N) { C[p.cOff + b * p.cBatch + m * p.ldc + n] = acc[i][j]; }
    }
  }
}`;

// Shared helpers for one-workgroup-per-row kernels (256 threads).
const rowReduce = `var<workgroup> red: array<f32, 256>;
fn sum256(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn max256(v: f32, li: u32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}`;

// Row-wise RMSNorm variants over rows of d <= 1024 (4 values per thread held in registers).
//  norm:      Y[r] = rms(X[r] * scale) * w            (w optional: HASW)
//  addnorm:   O[r] = rms(O[r]) * w;  H[r] += O[r];  Y[r] = rms(H[r]) * w2   (the residual add + next pre-norm)
//  addscale:  H[r] = (H[r] + rms(O[r]) * w) * scalar  (the per-layer-embedding block's end)
export const norm = (kind, hasW = true) => `${uniforms(['d', 'ldx', 'ldy', 'xOff', 'yOff', 'scaleBits'])}
@group(0) @binding(1) var<storage, read> W: array<f32>;
${kind === 'norm' ? `@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;`
  : kind === 'addnorm' ? `@group(0) @binding(2) var<storage, read> O: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> W2: array<f32>;`
  : `@group(0) @binding(2) var<storage, read> O: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
@group(0) @binding(4) var<storage, read> S: array<f32>;`}
${rowReduce}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * 65535u;
  let d = p.d;
  let eps = 1e-6;
  var v: array<f32, 4>;
  var ss = 0.0;
  ${kind === 'norm' ? `let sc = bitcast<f32>(p.scaleBits);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { v[i] = X[p.xOff + r * p.ldx + c] * sc; ss += v[i] * v[i]; }
  }
  let inv = pow(sum256(ss, li) / f32(d) + eps, -0.5);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { Y[p.yOff + r * p.ldy + c] = v[i] * inv${hasW ? ' * W[c]' : ''}; }
  }` : kind === 'addnorm' ? `for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { v[i] = O[r * d + c]; ss += v[i] * v[i]; }
  }
  let inv = pow(sum256(ss, li) / f32(d) + eps, -0.5);
  var ss2 = 0.0;
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { let h = H[r * d + c] + v[i] * inv * W[c]; H[r * d + c] = h; v[i] = h; ss2 += h * h; }
  }
  let inv2 = pow(sum256(ss2, li) / f32(d) + eps, -0.5);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { Y[r * d + c] = v[i] * inv2 * W2[c]; }
  }` : `for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { v[i] = O[r * d + c]; ss += v[i] * v[i]; }
  }
  let inv = pow(sum256(ss, li) / f32(d) + eps, -0.5);
  let s = S[0];
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { H[r * d + c] = (H[r * d + c] + v[i] * inv * W[c]) * s; }
  }`}
}`;

// Per-head q/k RMSNorm (weighted) + RoPE, v RMSNorm, in place in the [t, (nq + 2 nkv) * hd] qkv buffer. One workgroup
// per (token, head slot). AXIAL: the first half of each head rotates by the patch column (pos % gw), the second by the
// row (pos / gw), each with half = hd / 4; else 1-D with half = hd / 2. cos/sin tables are [pos, half].
export const qkvPrep = (axial) => `${uniforms(['hd', 'nq', 'nkv', 'gw'])}
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> QN: array<f32>;
@group(0) @binding(3) var<storage, read> KN: array<f32>;
@group(0) @binding(4) var<storage, read> COS: array<f32>;
@group(0) @binding(5) var<storage, read> SIN: array<f32>;
var<workgroup> xs: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let pos = wg.x;
  let slot = wg.y;
  let hd = p.hd;
  let row = (p.nq + 2u * p.nkv) * hd;
  let base = pos * row + slot * hd;
  var ss = 0.0;
  for (var c = li; c < hd; c += 128u) { let v = X[base + c]; xs[c] = v; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let inv = pow(red[0] / f32(hd) + 1e-6, -0.5);
  let isq = slot < p.nq;
  let isk = !isq && slot < p.nq + p.nkv;
  for (var c = li; c < hd; c += 128u) {
    var w = 1.0;
    if (isq) { w = QN[c]; } else if (isk) { w = KN[c]; }
    xs[c] = xs[c] * inv * w;
  }
  workgroupBarrier();
  for (var c = li; c < hd; c += 128u) {
    var y = xs[c];
    if (isq || isk) {
      ${axial ? `let seg = hd / 2u;
      let s0 = (c / seg) * seg;
      var ppos = pos % p.gw;
      if (s0 > 0u) { ppos = pos / p.gw; }
      let half = seg / 2u;
      let j = c - s0;` : `let s0 = 0u;
      let ppos = pos;
      let half = hd / 2u;
      let j = c;`}
      let i = j % half;
      let cs = COS[ppos * half + i];
      let sn = SIN[ppos * half + i];
      if (j < half) { y = xs[c] * cs - xs[c + half] * sn; } else { y = xs[c] * cs + xs[c - half] * sn; }
    }
    X[base + c] = y;
  }
}`;

// Row softmax over S rows of n keys, masked to |i - j| <= window (i = row % rowsPerHead), in place.
export const softmax = `${uniforms(['n', 'ld', 'rowsPerHead', 'window'])}
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
${rowReduce}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * 65535u;
  let qi = r % p.rowsPerHead;
  let base = r * p.ld;
  let lo = select(0u, qi - p.window, qi > p.window);
  let hi = min(p.n, qi + p.window + 1u);
  var m = -3.0e38;
  for (var j = li; j < p.n; j += 256u) { if (j >= lo && j < hi) { m = max(m, S[base + j]); } }
  m = max256(m, li);
  var s = 0.0;
  for (var j = li; j < p.n; j += 256u) {
    var e = 0.0;
    if (j >= lo && j < hi) { e = exp(S[base + j] - m); }
    S[base + j] = e;
    s += e;
  }
  let inv = 1.0 / sum256(s, li);
  for (var j = li; j < p.n; j += 256u) { S[base + j] *= inv; }
}`;

// Y[r, c] = gelu_tanh(A[r, c]) * B[r, c], with row strides and offsets.
export const geluMul = `${uniforms(['rows', 'n', 'lda', 'ldb', 'ldy', 'aOff', 'bOff'])}
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  let r = g.y;
  if (c >= p.n || r >= p.rows) { return; }
  let x = A[p.aOff + r * p.lda + c];
  let u = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  Y[r * p.ldy + c] = 0.5 * x * (1.0 + tanh(u)) * B[p.bOff + r * p.ldb + c];
}`;

// h[i, c] += pos[0, i % gw, c] + pos[1, i / gw, c], pos bf16 [2, rows, d].
export const addPos = `${uniforms(['t', 'd', 'gw', 'posRows'])}
@group(0) @binding(1) var<storage, read_write> H: array<f32>;
@group(0) @binding(2) var<storage, read> POS: array<u32>;
fn bf(i: u32) -> f32 {
  let w = POS[i / 2u];
  return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (i & 1u) == 1u));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  let i = g.y;
  if (c >= p.d || i >= p.t) { return; }
  let x = bf((i % p.gw) * p.d + c);
  let y = bf((p.posRows + i / p.gw) * p.d + c);
  H[i * p.d + c] += x + y;
}`;

// Patches: [t, 16*16*3] in (row, column, channel) order, scaled to [-1, 1], from RGB bytes packed in u32s.
export const patchify = `${uniforms(['gw', 'width', 't'])}
@group(0) @binding(1) var<storage, read> IMG: array<u32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let e = g.x;
  let i = g.y;
  if (e >= 768u || i >= p.t) { return; }
  let r = e / 48u;
  let rem = e % 48u;
  let py = i / p.gw;
  let px = i % p.gw;
  let byte = ((py * 16u + r) * p.width + px * 16u) * 3u + rem;
  let b = (IMG[byte / 4u] >> ((byte % 4u) * 8u)) & 255u;
  X[i * 768u + e] = 2.0 * (f32(b) * (1.0 / 255.0) - 0.5);
}`;

// 3x3 average pooling over the patch grid, * sqrt(d), RMSNorm without weight. One workgroup per output token.
export const pool = `${uniforms(['d', 'gw', 'ow', 'k'])}
@group(0) @binding(1) var<storage, read> H: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
${rowReduce}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let o = wg.x;
  let oy = o / p.ow;
  let ox = o % p.ow;
  let inv = 1.0 / f32(p.k * p.k);
  let sc = sqrt(f32(p.d));
  var v: array<f32, 4>;
  var ss = 0.0;
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < p.d) {
      var a = 0.0;
      for (var dy = 0u; dy < p.k; dy++) {
        for (var dx = 0u; dx < p.k; dx++) {
          a += H[((oy * p.k + dy) * p.gw + ox * p.k + dx) * p.d + c] * inv;
        }
      }
      v[i] = a * sc;
      ss += v[i] * v[i];
    }
  }
  let r = pow(sum256(ss, li) / f32(p.d) + 1e-6, -0.5);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < p.d) { Y[o * p.d + c] = v[i] * r; }
  }
}`;

// Mean over t rows of X [t, d] (already final-normed), times out_proj [e, d] (bf16), L2-normalized. One workgroup.
export const meanProj = `${uniforms(['t', 'd', 'e'])}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> m: array<f32, 1024>;
var<workgroup> outv: array<f32, 1024>;
${rowReduce}
fn bf(i: u32) -> f32 {
  let w = W[i / 2u];
  return bitcast<f32>(select(w << 16u, w & 0xffff0000u, (i & 1u) == 1u));
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  for (var c = li; c < p.d; c += 256u) {
    var a = 0.0;
    for (var r = 0u; r < p.t; r++) { a += X[r * p.d + c]; }
    m[c] = a / f32(p.t);
  }
  workgroupBarrier();
  var ss = 0.0;
  for (var j = li; j < p.e; j += 256u) {
    var a = 0.0;
    for (var c = 0u; c < p.d; c++) { a += m[c] * bf(j * p.d + c); }
    outv[j] = a;
    ss += a * a;
  }
  let n = max(sqrt(sum256(ss, li)), 1e-12);
  for (var j = li; j < p.e; j += 256u) { Y[j] = outv[j] / n; }
}`;

// Generic tiled C = A W^T with bf16 W: BM x BN tiles, BK deep, each thread a TM x TN block (TN % 4 == 0).
// SPLIT > 1: grid.z slices K; slice z writes C + z * sliceStride (partials for reduceK). MODE 1 adds into C.
export const matmulW2 = ({ BM, BN, TM, TN, BK = 16, mode = 0 }) => {
  const T = (BM / TM) * (BN / TN);
  const TNV = TN / 4;
  return `${uniforms(['M', 'N', 'K', 'lda', 'ldc', 'aOff', 'cOff', 'kSlice', 'sliceStride'])}
@group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> W: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;
const BM = ${BM}u; const BN = ${BN}u; const BK = ${BK}u; const TM = ${TM}u; const TN = ${TN}u; const T = ${T}u;
var<workgroup> As: array<f32, ${BM * BK}>;  // [k][m]
var<workgroup> BsS: array<f32, ${BN * BK}>;  // [k][n]
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  let tx = li % (BN / TN);
  let ty = li / (BN / TN);
  let kBeg = wg.z * p.kSlice;
  let kEnd = min(p.K, kBeg + p.kSlice);
  var acc: array<vec4<f32>, ${TM * TNV}>;
  for (var k0 = kBeg; k0 < kEnd; k0 += BK) {
    for (var e = li; e < BM * BK / 4u; e += T) {
      let r = e / (BK / 4u);
      let kq = (e % (BK / 4u)) * 4u;
      var av = vec4<f32>(0.0);
      if (m0 + r < p.M) { av = A[(p.aOff + (m0 + r) * p.lda + k0 + kq) / 4u]; }
      As[(kq + 0u) * BM + r] = av.x;
      As[(kq + 1u) * BM + r] = av.y;
      As[(kq + 2u) * BM + r] = av.z;
      As[(kq + 3u) * BM + r] = av.w;
    }
    // W: BN rows x BK k, as groups of 4 bf16 along k; scattered to Bs[k][n].
    for (var e = li; e < BN * BK / 4u; e += T) {
      let r = e / (BK / 4u);
      let kq = (e % (BK / 4u)) * 4u;
      let wv = W[((n0 + r) * p.K + k0 + kq) / 4u];
      let b0 = bitcast<f32>(wv.x << 16u);
      let b1 = bitcast<f32>(wv.x & 0xffff0000u);
      let b2 = bitcast<f32>(wv.y << 16u);
      let b3 = bitcast<f32>(wv.y & 0xffff0000u);
      BsS[(kq + 0u) * BN + r] = b0;
      BsS[(kq + 1u) * BN + r] = b1;
      BsS[(kq + 2u) * BN + r] = b2;
      BsS[(kq + 3u) * BN + r] = b3;
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var b: array<vec4<f32>, ${TNV}>;
      for (var j = 0u; j < ${TNV}u; j++) {
        let o = k * BN + tx * TN + j * 4u;
        b[j] = vec4<f32>(BsS[o], BsS[o + 1u], BsS[o + 2u], BsS[o + 3u]);
      }
      for (var i = 0u; i < TM; i++) {
        let a = As[k * BM + ty * TM + i];
        for (var j = 0u; j < ${TNV}u; j++) { acc[i * ${TNV}u + j] += a * b[j]; }
      }
    }
    workgroupBarrier();
  }
  let cBase = p.cOff / 4u + wg.z * (p.sliceStride / 4u);
  for (var i = 0u; i < TM; i++) {
    let m = m0 + ty * TM + i;
    if (m < p.M) {
      for (var j = 0u; j < ${TNV}u; j++) {
        let c = cBase + (m * p.ldc + n0 + tx * TN + j * 4u) / 4u;
        ${mode === 1 ? 'C[c] += acc[i * ' + TNV + 'u + j];' : 'C[c] = acc[i * ' + TNV + 'u + j];'}
      }
    }
  }
}`;
};

// C[m, n] (+)= sum_z P[z][m, n], P slices [splits, M, N] contiguous; C with row stride ldc and offset cOff.
export const reduceK = (mode = 0) => `${uniforms(['M', 'N', 'splits', 'ldc', 'cOff'])}
@group(0) @binding(1) var<storage, read> Pt: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  let nv = p.N / 4u;
  if (i >= p.M * nv) { return; }
  let m = i / nv;
  let n = i % nv;
  var s = vec4<f32>(0.0);
  for (var z = 0u; z < p.splits; z++) { s += Pt[z * p.M * nv + i]; }
  let c = (p.cOff + m * p.ldc) / 4u + n;
  ${mode === 1 ? 'C[c] += s;' : 'C[c] = s;'}
}`;

// C = A W^T on subgroup matrices (chromium_experimental_subgroup_matrix, f32 8x8x8): BM x BN tiles, BK deep,
// 4 subgroups of 32 each owning a (BM/2) x (BN/2) block. C's rows must be padded to a multiple of BM (stores
// are whole 8x8 blocks); A rows >= M load as zero. grid.z slices K as in matmulW2.
export const matmulSG = ({ BM = 32, BN = 32, BK = 16, PAD = 0, GA = 0 } = {}) => {
  const SK = BK + PAD;  // shared row stride
  // One subgroup (32 invocations) per workgroup, so every matrix offset derives from workgroup_id (uniform).
  const FI = BM / 8, FJ = BN / 8;
  const accs = [];
  for (let i = 0; i < FI; i++) for (let j = 0; j < FJ; j++) accs.push([i, j]);
  return `enable chromium_experimental_subgroup_matrix;
${uniforms(['M', 'N', 'K', 'lda', 'ldc', 'aOff', 'cOff', 'kSlice', 'sliceStride'])}
@group(0) @binding(1) var<storage, read> A: array<${GA ? 'f32' : 'vec4<f32>'}>;
@group(0) @binding(2) var<storage, read> W: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
const BM = ${BM}u; const BN = ${BN}u; const BK = ${BK}u; const SK = ${SK}u;
${GA ? '' : `var<workgroup> As: array<f32, ${BM * SK}>;  // [m][k]`}
var<workgroup> Bs: array<f32, ${BN * SK}>;  // [n][k]
alias L = subgroup_matrix_left<f32, 8, 8>;
alias R = subgroup_matrix_right<f32, 8, 8>;
alias Res = subgroup_matrix_result<f32, 8, 8>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  let kBeg = wg.z * p.kSlice;
  let kEnd = min(p.K, kBeg + p.kSlice);
  ${accs.map(([i, j]) => `var a${i}_${j}: Res;`).join('\n  ')}
  for (var k0 = kBeg; k0 < kEnd; k0 += BK) {
    ${GA ? '' : `for (var e = li; e < BM * BK / 4u; e += 32u) {
      let r = e / (BK / 4u);
      let kq = (e % (BK / 4u)) * 4u;
      var av = vec4<f32>(0.0);
      if (m0 + r < p.M) { av = A[(p.aOff + (m0 + r) * p.lda + k0 + kq) / 4u]; }
      let o = r * SK + kq;
      As[o] = av.x; As[o + 1u] = av.y; As[o + 2u] = av.z; As[o + 3u] = av.w;
    }`}
    for (var e = li; e < BN * BK / 4u; e += 32u) {
      let r = e / (BK / 4u);
      let kq = (e % (BK / 4u)) * 4u;
      let wv = W[((n0 + r) * p.K + k0 + kq) / 4u];
      let o = r * SK + kq;
      Bs[o] = bitcast<f32>(wv.x << 16u);
      Bs[o + 1u] = bitcast<f32>(wv.x & 0xffff0000u);
      Bs[o + 2u] = bitcast<f32>(wv.y << 16u);
      Bs[o + 3u] = bitcast<f32>(wv.y & 0xffff0000u);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk += 8u) {
      ${Array.from({ length: FI }, (_, i) => (GA ? `let l${i} = subgroupMatrixLoad<L>(&A, p.aOff + (m0 + ${i * 8}u) * p.lda + k0 + kk, false, p.lda);` : `let l${i} = subgroupMatrixLoad<L>(&As, ${i * 8}u * SK + kk, false, SK);`)).join('\n      ')}
      ${Array.from({ length: FJ }, (_, j) => `let r${j} = subgroupMatrixLoad<R>(&Bs, ${j * 8}u * SK + kk, true, SK);`).join('\n      ')}
      ${accs.map(([i, j]) => `a${i}_${j} = subgroupMatrixMultiplyAccumulate(l${i}, r${j}, a${i}_${j});`).join('\n      ')}
    }
    workgroupBarrier();
  }
  let cBase = p.cOff + wg.z * p.sliceStride;
  ${accs.map(([i, j]) => `subgroupMatrixStore(&C, cBase + (m0 + ${i * 8}u) * p.ldc + n0 + ${j * 8}u, a${i}_${j}, false, p.ldc);`).join('\n  ')}
}`;
};

// Fused full (unmasked) attention for head_dim 64, reading q/k/v from the [t, 3 nh 64] qkv buffer, online softmax
// over key tiles of 32, writing out [t, nh 64]. One workgroup (128 threads) per (32-query block, head). Each run of
// 8 consecutive invocations owns 2 query rows (scores for 4 keys each, outputs for 8 columns each); row max and sum
// reduce across those 8 lanes with subgroup shuffles (subgroup size >= 8, invocations laid out linearly). Scale 1.
export const flash64 = `enable subgroups;
${uniforms(['t', 'nh'])}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
const BQ = 32u; const BKV = 32u; const HD = 64u;
var<workgroup> Qt: array<vec2<f32>, 1024>;  // [c][q/2]
var<workgroup> Kt: array<vec4<f32>, 512>;   // [c][k/4]
var<workgroup> Vs: array<vec4<f32>, 512>;   // [k][c/4]
var<workgroup> Ps: array<vec4<f32>, 256>;   // [q][k/4]
fn rmax(v: f32) -> f32 {
  var m = max(v, subgroupShuffleXor(v, 1u));
  m = max(m, subgroupShuffleXor(m, 2u));
  return max(m, subgroupShuffleXor(m, 4u));
}
fn rsum(v: f32) -> f32 {
  var m = v + subgroupShuffleXor(v, 1u);
  m = m + subgroupShuffleXor(m, 2u);
  return m + subgroupShuffleXor(m, 4u);
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let q0 = wg.x * BQ;
  let h = wg.y;
  let row = 3u * p.nh * HD;
  for (var e = li; e < BQ * HD / 2u; e += 128u) {
    let qp = e / HD; let c = e % HD;  // query pair qp, column c
    let qa = q0 + 2u * qp;
    var v = vec2<f32>(0.0);
    if (qa < p.t) { v.x = X[qa * row + h * HD + c]; }
    if (qa + 1u < p.t) { v.y = X[(qa + 1u) * row + h * HD + c]; }
    Qt[c * 16u + qp] = v;
  }
  let r = li / 8u;          // row pair: queries 2r, 2r+1
  let kq = li % 8u;         // keys 4kq..4kq+3 in the tile
  let oc = kq * 2u;         // output vec4 columns oc, oc+1 (8 columns)
  var m = vec2<f32>(-3.0e38);
  var l = vec2<f32>(0.0);
  var o0: array<vec4<f32>, 2>;  // row 2r
  var o1: array<vec4<f32>, 2>;  // row 2r+1
  for (var k0 = 0u; k0 < p.t; k0 += BKV) {
    workgroupBarrier();
    for (var e = li; e < BKV * HD / 4u; e += 128u) {
      // K: (c, key quad); V: (key, column quad)
      let c = e / 8u; let kk = (e % 8u) * 4u;
      var kv = vec4<f32>(0.0);
      for (var z = 0u; z < 4u; z++) {
        if (k0 + kk + z < p.t) { kv[z] = X[(k0 + kk + z) * row + (p.nh + h) * HD + c]; }
      }
      Kt[c * 8u + kk / 4u] = kv;
      let k = e / 16u; let cq = (e % 16u) * 4u;
      var vv = vec4<f32>(0.0);
      if (k0 + k < p.t) {
        let b = (k0 + k) * row + (2u * p.nh + h) * HD + cq;
        vv = vec4<f32>(X[b], X[b + 1u], X[b + 2u], X[b + 3u]);
      }
      Vs[k * 16u + cq / 4u] = vv;
    }
    workgroupBarrier();
    var sa = vec4<f32>(0.0);
    var sb = vec4<f32>(0.0);
    for (var c = 0u; c < HD; c++) {
      let q = Qt[c * 16u + r];
      let kk = Kt[c * 8u + kq];
      sa += q.x * kk;
      sb += q.y * kk;
    }
    for (var z = 0u; z < 4u; z++) {
      if (k0 + kq * 4u + z >= p.t) { sa[z] = -3.0e38; sb[z] = -3.0e38; }
    }
    let mx = vec2<f32>(max(m.x, rmax(max(max(sa.x, sa.y), max(sa.z, sa.w)))),
                       max(m.y, rmax(max(max(sb.x, sb.y), max(sb.z, sb.w)))));
    let pa = exp(sa - mx.x);
    let pb = exp(sb - mx.y);
    let alpha = exp(m - mx);
    l = l * alpha + vec2<f32>(rsum(pa.x + pa.y + pa.z + pa.w), rsum(pb.x + pb.y + pb.z + pb.w));
    m = mx;
    Ps[(2u * r) * 8u + kq] = pa;
    Ps[(2u * r + 1u) * 8u + kq] = pb;
    workgroupBarrier();
    o0[0] *= alpha.x; o0[1] *= alpha.x;
    o1[0] *= alpha.y; o1[1] *= alpha.y;
    for (var k4 = 0u; k4 < 8u; k4++) {
      let pa4 = Ps[(2u * r) * 8u + k4];
      let pb4 = Ps[(2u * r + 1u) * 8u + k4];
      for (var z = 0u; z < 4u; z++) {
        let k = k4 * 4u + z;
        let v0 = Vs[k * 16u + oc];
        let v1 = Vs[k * 16u + oc + 1u];
        o0[0] += pa4[z] * v0; o0[1] += pa4[z] * v1;
        o1[0] += pb4[z] * v0; o1[1] += pb4[z] * v1;
      }
    }
  }
  let qa = q0 + 2u * r;
  let base = h * HD + oc * 4u;
  if (qa < p.t) {
    let b = qa * p.nh * HD + base;
    let i0 = o0[0] / l.x; let i1 = o0[1] / l.x;
    O[b] = i0.x; O[b + 1u] = i0.y; O[b + 2u] = i0.z; O[b + 3u] = i0.w;
    O[b + 4u] = i1.x; O[b + 5u] = i1.y; O[b + 6u] = i1.z; O[b + 7u] = i1.w;
  }
  if (qa + 1u < p.t) {
    let b = (qa + 1u) * p.nh * HD + base;
    let i0 = o1[0] / l.y; let i1 = o1[1] / l.y;
    O[b] = i0.x; O[b + 1u] = i0.y; O[b + 2u] = i0.z; O[b + 3u] = i0.w;
    O[b + 4u] = i1.x; O[b + 5u] = i1.y; O[b + 6u] = i1.z; O[b + 7u] = i1.w;
  }
}`;

// Axial q/k norm + 2-D RoPE and v norm for head_dim 64: 4 heads per 256-thread workgroup, one thread per element,
// sums over each head's 64 values through shared memory.
export const qkvAxial64 = `${uniforms(['nh', 'gw', 'slots'])}
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> QN: array<f32>;
@group(0) @binding(3) var<storage, read> KN: array<f32>;
@group(0) @binding(4) var<storage, read> COS: array<f32>;
@group(0) @binding(5) var<storage, read> SIN: array<f32>;
var<workgroup> xs: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let g = li / 64u;            // head within the workgroup
  let c = li % 64u;
  let gslot = wg.x * 4u + g;   // global (pos, slot) index
  let pos = gslot / p.slots;
  let slot = gslot % p.slots;
  let idx = gslot * 64u + c;   // rows are slots * 64 wide, so this is pos * row + slot * 64 + c
  let x = X[idx];
  red[li] = x * x;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (c < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let inv = pow(red[g * 64u] / 64.0 + 1e-6, -0.5);
  let isq = slot < p.nh;
  let isk = !isq && slot < 2u * p.nh;
  var w = 1.0;
  if (isq) { w = QN[c]; } else if (isk) { w = KN[c]; }
  let y = x * inv * w;
  xs[li] = y;
  workgroupBarrier();
  var outv = y;
  if (isq || isk) {
    let s0 = (c / 32u) * 32u;
    var ppos = pos % p.gw;
    if (s0 > 0u) { ppos = pos / p.gw; }
    let j = c - s0;
    let i = j % 16u;
    let cs = COS[ppos * 16u + i];
    let sn = SIN[ppos * 16u + i];
    if (j < 16u) { outv = y * cs - xs[li + 16u] * sn; } else { outv = y * cs + xs[li - 16u] * sn; }
  }
  X[idx] = outv;
}`;
