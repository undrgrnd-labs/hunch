// eg2/kernels.js
var uniforms = (fields) => `struct P { ${fields.map((f) => `${f}: u32`).join(", ")} }
@group(0) @binding(0) var<uniform> p: P;`;
var matmulF32 = (transB) => `${uniforms(["M", "N", "K", "lda", "ldb", "ldc", "aOff", "bOff", "cOff", "aBatch", "bBatch", "cBatch", "bDiv"])}
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
      ${transB ? `var w = 0.0;
      if (n0 + r < p.N && k0 + kk < p.K) { w = B[bBase + (n0 + r) * p.ldb + k0 + kk]; }
      Bs[kk][r] = w;` : `let kk2 = idx / 64u;
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
var rowReduce = `var<workgroup> red: array<f32, 256>;
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
var norm = (kind, hasW = true) => `${uniforms(["d", "ldx", "ldy", "xOff", "yOff", "scaleBits"])}
@group(0) @binding(1) var<storage, read> W: array<f32>;
${kind === "norm" ? `@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;` : kind === "addnorm" ? `@group(0) @binding(2) var<storage, read> O: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> W2: array<f32>;` : `@group(0) @binding(2) var<storage, read> O: array<f32>;
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
  ${kind === "norm" ? `let sc = bitcast<f32>(p.scaleBits);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { v[i] = X[p.xOff + r * p.ldx + c] * sc; ss += v[i] * v[i]; }
  }
  let inv = pow(sum256(ss, li) / f32(d) + eps, -0.5);
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 256u;
    if (c < d) { Y[p.yOff + r * p.ldy + c] = v[i] * inv${hasW ? " * W[c]" : ""}; }
  }` : kind === "addnorm" ? `for (var i = 0u; i < 4u; i++) {
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
var qkvPrep = (axial) => `${uniforms(["hd", "nq", "nkv", "gw"])}
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
var softmax = `${uniforms(["n", "ld", "rowsPerHead", "window"])}
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
var geluMul = `${uniforms(["rows", "n", "lda", "ldb", "ldy", "aOff", "bOff"])}
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
var addPos = `${uniforms(["t", "d", "gw", "posRows"])}
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
var patchify = `${uniforms(["gw", "width", "t"])}
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
var pool = `${uniforms(["d", "gw", "ow", "k"])}
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
var meanProj = `${uniforms(["t", "d", "e"])}
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
var matmulW2 = ({ BM, BN, TM, TN, BK = 16, mode = 0 }) => {
  const T = BM / TM * (BN / TN);
  const TNV = TN / 4;
  return `${uniforms(["M", "N", "K", "lda", "ldc", "aOff", "cOff", "kSlice", "sliceStride"])}
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
        ${mode === 1 ? "C[c] += acc[i * " + TNV + "u + j];" : "C[c] = acc[i * " + TNV + "u + j];"}
      }
    }
  }
}`;
};
var reduceK = (mode = 0) => `${uniforms(["M", "N", "splits", "ldc", "cOff"])}
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
  ${mode === 1 ? "C[c] += s;" : "C[c] = s;"}
}`;
var matmulSG = ({ BM = 32, BN = 32, BK = 16, PAD = 0, GA = 0 } = {}) => {
  const SK = BK + PAD;
  const FI = BM / 8, FJ = BN / 8;
  const accs = [];
  for (let i = 0; i < FI; i++) for (let j = 0; j < FJ; j++) accs.push([i, j]);
  return `enable chromium_experimental_subgroup_matrix;
${uniforms(["M", "N", "K", "lda", "ldc", "aOff", "cOff", "kSlice", "sliceStride"])}
@group(0) @binding(1) var<storage, read> A: array<${GA ? "f32" : "vec4<f32>"}>;
@group(0) @binding(2) var<storage, read> W: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
const BM = ${BM}u; const BN = ${BN}u; const BK = ${BK}u; const SK = ${SK}u;
${GA ? "" : `var<workgroup> As: array<f32, ${BM * SK}>;  // [m][k]`}
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
  ${accs.map(([i, j]) => `var a${i}_${j}: Res;`).join("\n  ")}
  for (var k0 = kBeg; k0 < kEnd; k0 += BK) {
    ${GA ? "" : `for (var e = li; e < BM * BK / 4u; e += 32u) {
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
      ${Array.from({ length: FI }, (_, i) => GA ? `let l${i} = subgroupMatrixLoad<L>(&A, p.aOff + (m0 + ${i * 8}u) * p.lda + k0 + kk, false, p.lda);` : `let l${i} = subgroupMatrixLoad<L>(&As, ${i * 8}u * SK + kk, false, SK);`).join("\n      ")}
      ${Array.from({ length: FJ }, (_, j) => `let r${j} = subgroupMatrixLoad<R>(&Bs, ${j * 8}u * SK + kk, true, SK);`).join("\n      ")}
      ${accs.map(([i, j]) => `a${i}_${j} = subgroupMatrixMultiplyAccumulate(l${i}, r${j}, a${i}_${j});`).join("\n      ")}
    }
    workgroupBarrier();
  }
  let cBase = p.cOff + wg.z * p.sliceStride;
  ${accs.map(([i, j]) => `subgroupMatrixStore(&C, cBase + (m0 + ${i * 8}u) * p.ldc + n0 + ${j * 8}u, a${i}_${j}, false, p.ldc);`).join("\n  ")}
}`;
};
var flash64 = `enable subgroups;
${uniforms(["t", "nh"])}
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
var qkvAxial64 = `${uniforms(["nh", "gw", "slots"])}
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

// eg2/model.js
var PATCH = 16;
var POOL = 3;
async function load(base = ".", { subgroupMatrix = true, onProgress = null } = {}) {
  const t0 = performance.now();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension
    },
    requiredFeatures: ["timestamp-query", "subgroups", "shader-f16", "chromium-experimental-subgroup-matrix"].filter((f) => adapter.features.has(f))
  });
  device.lost.then((i) => console.error("device lost", i.message));
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
  const special = new Float32Array(bin, man.tensors["t.special"].offset, 4 * 512);
  const m = new Model(device, adapter, man, W, special.slice());
  m.useSG = subgroupMatrix && device.features.has("chromium-experimental-subgroup-matrix");
  await device.queue.onSubmittedWorkDone();
  m.loadMs = { total: performance.now() - t0, fetch: tFetch };
  return m;
}
var f32bits = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!onProgress || !total || !res.body) return res.arrayBuffer();
  const buf = new Uint8Array(total);
  let at = 0;
  for (const reader = res.body.getReader(); ; ) {
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
  const inv = Array.from({ length: half }, (_, i) => Math.fround(1 / Math.pow(theta, Math.fround(2 * i / dim))));
  const cos = new Float32Array(t * half), sin = new Float32Array(t * half);
  for (let p = 0; p < t; p++) for (let i = 0; i < half; i++) {
    const a = Math.fround(p * inv[i]);
    cos[p * half + i] = Math.cos(a);
    sin[p * half + i] = Math.sin(a);
  }
  return { cos, sin };
}
var Model = class {
  constructor(device, adapter, man, W, special) {
    this.device = device;
    this.adapter = adapter;
    this.man = man;
    this.W = W;
    this.vc = man.config.vision_config;
    this.tc = man.config.text_config;
    this.special = special;
    this.pipes = /* @__PURE__ */ new Map();
    this.plans = /* @__PURE__ */ new Map();
  }
  pipe(key, code) {
    let p = this.pipes.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
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
      const sg = { BM: 32, BN: 32, BK: 16 };
      this.op(
        plan,
        `mmSG_${JSON.stringify(sg)}`,
        matmulSG(sg),
        [M, N, Kd, lda, ldc, aOff, cOff, Kd, 0],
        [A, Wt, C],
        [N / sg.BN, Math.ceil(M / sg.BM), 1],
        label
      );
      return;
    }
    if (M > 128) cfg = { BM: 64, BN: 64, TM: 8, TN: 4 };
    else {
      const BM = Math.ceil(M / 16) * 16;
      cfg = { BM, BN: 64, TM: BM / 16, TN: 4 };
      const tiles = N / 64;
      while (split < 8 && tiles * split < 128 && Kd / (split * 2) % 16 === 0 && Kd / (split * 2) >= 64) split *= 2;
    }
    const groups = [N / cfg.BN, Math.ceil(M / cfg.BM), split];
    const kslice = Kd / split;
    const key = `mmW2_${JSON.stringify(cfg)}_${add && split === 1 ? 1 : 0}`;
    const code = matmulW2({ ...cfg, mode: add && split === 1 ? 1 : 0 });
    if (split === 1) {
      this.op(plan, key, code, [M, N, Kd, lda, ldc, aOff, cOff, kslice, 0], [A, Wt, C], groups, label);
      return;
    }
    if (split * M * N > plan.partialFloats) throw new Error("partial buffer too small");
    this.op(plan, key, code, [M, N, Kd, lda, N, aOff, 0, kslice, M * N], [A, Wt, plan.B.partial], groups, label);
    const nv = M * N / 4;
    this.op(
      plan,
      `reduceK${add ? 1 : 0}`,
      reduceK(add ? 1 : 0),
      [M, N, split, ldc, cOff],
      [plan.B.partial, C],
      [Math.min(Math.ceil(nv / 256), 65535), Math.ceil(nv / 256 / 65535), 1],
      label + ":reduce"
    );
  }
  mmF(plan, transB, A, B, C, u, groups) {
    this.op(plan, `mmF${transB}`, matmulF32(transB), u, [A, B, C], groups);
  }
  // Multi-head attention from a [t, (nq + 2 nkv) hd] qkv buffer into out [t, nq hd], chunked over heads so S stays
  // under ~64 MB.
  attention(plan, qkv, out, S, t, nq, nkv, hd, window) {
    const row = (nq + 2 * nkv) * hd;
    const per = Math.max(1, Math.min(nq, Math.floor(this.sFloats / (t * t))));
    const grp = nq / nkv;
    for (let h0 = 0; h0 < nq; h0 += per) {
      const nh = Math.min(per, nq - h0);
      const g0 = Math.floor(h0 / grp);
      if (h0 % grp && per > 1) throw new Error("head chunk not aligned to KV groups");
      this.mmF(
        plan,
        true,
        qkv,
        qkv,
        S,
        [t, t, hd, row, row, t, h0 * hd, (nq + g0) * hd, 0, hd, hd, t * t, grp],
        [Math.ceil(t / 64), Math.ceil(t / 64), nh]
      );
      const rows = nh * t;
      this.op(plan, "softmax", softmax, [t, t, t, window], [S], [Math.min(rows, 65535), Math.ceil(rows / 65535), 1]);
      this.mmF(
        plan,
        false,
        S,
        qkv,
        out,
        [t, hd, t, t, row, nq * hd, 0, (nq + nkv + g0) * hd, h0 * hd, t * t, hd, hd, grp],
        [Math.ceil(hd / 64), Math.ceil(t / 64), nh]
      );
    }
  }
  // The dispatch list for one image size, with its activation buffers.
  plan(height, width) {
    const key = `${height}x${width}`;
    if (this.plans.has(key)) return this.plans.get(key);
    const { vc, tc, W } = this;
    const gh = height / PATCH, gw = width / PATCH, t = gh * gw;
    const d = vc.hidden_size, nh = vc.num_attention_heads, hd = vc.head_dim, inter = vc.intermediate_size;
    const n = gh / POOL * (gw / POOL), td = tc.hidden_size, tt = n + 4, pd = tc.hidden_size_per_layer_input;
    const nl = tc.num_hidden_layers;
    if (gh % POOL || gw % POOL || Math.max(gh, gw) > this.man.pos_rows) throw new Error(`bad image size ${key}`);
    this.sFloats = 16 * 1024 * 1024;
    const plan = { key, steps: [], marks: [], gh, gw, t, n, tt };
    const mark = (label, buffer, rows, cols) => plan.marks.push({ label, step: plan.steps.length, buffer, rows, cols });
    const tp = Math.ceil(t / 32) * 32;
    const B = {
      img: this.buf(Math.ceil(height * width * 3 / 4)),
      patches: this.buf(tp * 768),
      h: this.buf(tp * d),
      x: this.buf(tp * d),
      qkv: this.buf(tp * 3 * d),
      attn: this.buf(tp * d),
      o: this.buf(tp * d),
      gu: this.buf(tp * 2 * inter),
      g: this.buf(tp * inter),
      S: this.buf(this.sFloats),
      pooled: this.buf(n * d),
      partial: this.buf(8 * (tt + 16) * 4096),
      th: this.buf(tt * td),
      tin: this.buf(tt * td),
      tx: this.buf(tt * td),
      tqkv: this.buf(tt * 3072),
      tattn: this.buf(tt * 2048),
      to: this.buf(tt * td),
      tgu: this.buf(tt * 4096),
      tg: this.buf(tt * 2048),
      ple: this.buf(tt * nl * pd),
      pleRaw: this.buf(tt * nl * pd),
      pg: this.buf(tt * pd),
      out: this.buf(tc.embedding_dim)
    };
    plan.B = B;
    plan.partialFloats = 8 * (tt + 16) * 4096;
    plan.readback = this.device.createBuffer({ size: tc.embedding_dim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const vr = ropeTables(vc.rope_parameters.rope_theta, hd / 2, Math.max(gh, gw));
    const vcos = this.data(vr.cos), vsin = this.data(vr.sin);
    this.op(plan, "patchify", patchify, [gw, width, t], [B.img, B.patches], [3, t, 1]);
    mark("patches", B.patches, t, 768);
    this.mmW(plan, B.patches, W["v.input_proj"], B.h, t, d, 768, { padded: true });
    mark("v_inproj", B.h, t, d);
    this.op(plan, "addPos", addPos, [t, d, gw, this.man.pos_rows], [B.h, W["v.pos"]], [Math.ceil(d / 256), t, 1]);
    mark("v_embed", B.h, t, d);
    const rowsGrid = (r) => [Math.min(r, 65535), Math.ceil(r / 65535), 1];
    this.op(plan, "norm", norm("norm"), [d, d, d, 0, 0, f32bits(1)], [W["v0.input_layernorm"], B.h, B.x], rowsGrid(t));
    for (let L = 0; L < vc.num_hidden_layers; L++) {
      const P = `v${L}.`;
      this.mmW(plan, B.x, W[P + "wqkv"], B.qkv, t, 3 * d, d, { padded: true });
      if (hd !== 64) throw new Error("vision kernels assume head_dim 64");
      this.op(
        plan,
        "qkvAxial64",
        qkvAxial64,
        [nh, gw, 3 * nh],
        [B.qkv, W[P + "self_attn.q_norm"], W[P + "self_attn.k_norm"], vcos, vsin],
        [t * 3 * nh / 4, 1, 1]
      );
      this.op(plan, "flash64", flash64, [t, nh], [B.qkv, B.attn], [Math.ceil(t / 32), nh, 1]);
      this.mmW(plan, B.attn, W[P + "wo"], B.o, t, d, d, { padded: true });
      this.op(plan, "addnorm", norm("addnorm"), [d], [W[P + "post_attention_layernorm"], B.o, B.h, B.x, W[P + "pre_feedforward_layernorm"]], rowsGrid(t));
      this.mmW(plan, B.x, W[P + "wgu"], B.gu, t, 2 * inter, d, { padded: true });
      this.op(plan, "gelu", geluMul, [t, inter, 2 * inter, 2 * inter, inter, 0, inter], [B.gu, B.gu, B.g], [Math.ceil(inter / 256), t, 1]);
      this.mmW(plan, B.g, W[P + "wd"], B.o, t, d, inter, { padded: true });
      const next = L + 1 < vc.num_hidden_layers ? W[`v${L + 1}.input_layernorm`] : W[P + "input_layernorm"];
      this.op(plan, "addnorm", norm("addnorm"), [d], [W[P + "post_feedforward_layernorm"], B.o, B.h, B.x, next], rowsGrid(t));
      mark(`v_layer${L}`, B.h, t, d);
    }
    this.op(plan, "pool", pool, [d, gw, gw / POOL, POOL], [B.h, B.pooled], [n, 1, 1]);
    this.mmW(plan, B.pooled, W["v.proj"], B.th, n, td, d, { cOff: 2 * td });
    plan.textStart = plan.steps.length;
    mark("soft", B.th, tt, td);
    const sp = new Float32Array(4 * td);
    const es = Math.fround(Math.sqrt(td));
    for (let i = 0; i < sp.length; i++) sp[i] = Math.fround(this.special[i] * es);
    plan.special = this.data(sp);
    this.mmW(plan, B.tin, W["t.ple_model_proj"], B.pleRaw, tt, nl * pd, td);
    this.op(
      plan,
      "norm",
      norm("norm"),
      [pd, pd, pd, 0, 0, f32bits(Math.fround(Math.pow(td, -0.5)))],
      [W["t.ple_proj_norm"], B.pleRaw, B.ple],
      rowsGrid(tt * nl)
    );
    const tcos = {}, tsin = {};
    for (const [kind, dim] of [["sliding_attention", tc.head_dim], ["full_attention", 512]]) {
      const r = ropeTables(tc.rope_parameters[kind].rope_theta, dim, tt);
      tcos[kind] = this.data(r.cos);
      tsin[kind] = this.data(r.sin);
    }
    this.op(plan, "norm", norm("norm"), [td, td, td, 0, 0, f32bits(1)], [W["t0.input_layernorm"], B.th, B.tx], rowsGrid(tt));
    const tq = tc.num_attention_heads;
    for (let L = 0; L < nl; L++) {
      const P = `t${L}.`;
      const kind = tc.layer_types[L];
      const o = tc.per_layer_config[String(L).padStart(2, "0")] || {};
      const lhd = o.head_dim || tc.head_dim, kvh = o.num_key_value_heads || tc.num_key_value_heads;
      const qd = tq * lhd, row = (tq + 2 * kvh) * lhd;
      this.mmW(plan, B.tx, W[P + "wqkv"], B.tqkv, tt, row, td);
      this.op(
        plan,
        "qkv1d",
        qkvPrep(false),
        [lhd, tq, kvh, 1],
        [B.tqkv, W[P + "self_attn.q_norm"], W[P + "self_attn.k_norm"], tcos[kind], tsin[kind]],
        [tt, tq + 2 * kvh, 1]
      );
      const window = kind === "full_attention" ? 1 << 30 : tc.sliding_window;
      this.attention(plan, B.tqkv, B.tattn, B.S, tt, tq, kvh, lhd, window);
      this.mmW(plan, B.tattn, W[P + "wo"], B.to, tt, td, qd);
      this.op(plan, "addnorm", norm("addnorm"), [td], [W[P + "post_attention_layernorm"], B.to, B.th, B.tx, W[P + "pre_feedforward_layernorm"]], rowsGrid(tt));
      const ti = tc.intermediate_size;
      this.mmW(plan, B.tx, W[P + "wgu"], B.tgu, tt, 2 * ti, td);
      this.op(plan, "gelu", geluMul, [tt, ti, 2 * ti, 2 * ti, ti, 0, ti], [B.tgu, B.tgu, B.tg], [Math.ceil(ti / 256), tt, 1]);
      this.mmW(plan, B.tg, W[P + "wd"], B.to, tt, td, ti);
      this.op(plan, "addnorm", norm("addnorm"), [td], [W[P + "post_feedforward_layernorm"], B.to, B.th, B.tx, W[P + "pre_feedforward_layernorm"]], rowsGrid(tt));
      this.mmW(plan, B.th, W[P + "ple_gate"], B.pg, tt, pd, td);
      this.op(plan, "gelu", geluMul, [tt, pd, pd, nl * pd, pd, 0, L * pd], [B.pg, B.ple, B.tg], [Math.ceil(pd / 256), tt, 1]);
      this.mmW(plan, B.tg, W[P + "ple_proj"], B.to, tt, td, pd);
      this.op(plan, "addscale", norm("addscale"), [td], [W[P + "ple_block.post_per_layer_input_norm"], B.to, B.th, W[P + "layer_scalar"]], rowsGrid(tt));
      mark(`t_layer${L}`, B.th, tt, td);
      const next = L + 1 < nl ? W[`t${L + 1}.input_layernorm`] : W["t.norm"];
      this.op(plan, "norm", norm("norm"), [td, td, td, 0, 0, f32bits(1)], [next, B.th, B.tx], rowsGrid(tt));
    }
    this.op(plan, "meanProj", meanProj, [tt, td, tc.embedding_dim], [B.tx, W["t.out_proj"], B.out], [1, 1, 1]);
    mark("embedding", B.out, 1, tc.embedding_dim);
    this.plans.set(key, plan);
    return plan;
  }
  // Encode steps [0, upto) into one command buffer (one compute pass per dispatch run).
  encode(plan, upto = plan.steps.length, timestamps = null) {
    const enc = this.device.createCommandEncoder();
    const td = this.tc.hidden_size;
    let pass = null;
    const flush = () => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };
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
    const qs = this.device.createQuerySet({ type: "timestamp", count: 2 * n });
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
      pass.setPipeline(s.pipeline);
      pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(...s.groups);
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
      const k = (i < plan.textStart ? "v:" : "t:") + plan.steps[i].key + (plan.steps[i].label ? ":" + plan.steps[i].label : "");
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
};
function targetSize(h, w, maxPatches) {
  const targetPx = maxPatches * PATCH * PATCH;
  const factor = Math.sqrt(targetPx / (h * w));
  const side = POOL * PATCH;
  let th = Math.floor(factor * h / side) * side, tw = Math.floor(factor * w / side) * side;
  const maxSide = Math.floor(maxPatches / (POOL * POOL)) * side;
  if (th === 0 && tw === 0) throw new Error("image too small");
  if (th === 0) {
    th = side;
    tw = Math.min(Math.floor(w / h) * side, maxSide);
  } else if (tw === 0) {
    tw = side;
    th = Math.min(Math.floor(h / w) * side, maxSide);
  }
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
  return [rows.map(([xmin, w]) => [xmin, Int32Array.from(w, (v) => {
    v *= 2 ** prec;
    return v < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v);
  })]), prec];
}
function resizeAxis(src, outer, n, inner, out) {
  const [ws, prec] = resizeWeights(n, out);
  const round = 1 << prec - 1;
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
function resize(rgb, h, w, th, tw) {
  let data = rgb;
  if (tw !== w) data = resizeAxis(data, h, w, 3, tw);
  if (th !== h) data = resizeAxis(data, 1, h, tw * 3, th);
  return data;
}

// live.ts
var ELSE = "something else";
var dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
var norm2 = (a) => Math.sqrt(dot(a, a));
var unit = (v) => {
  const n = norm2(v) || 1;
  return Float32Array.from(v, (x) => x / n);
};
var sub = (a, b) => Float32Array.from(a, (x, i) => x - b[i]);
var sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
function wilson(k, m, z = 1.96) {
  if (m === 0) return [0, 1];
  const p = k / m, c = (p + z * z / (2 * m)) / (1 + z * z / m);
  const h = z * Math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m);
  return [c - h, c + h];
}
function softmax2(z) {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
var Eyes = class _Eyes {
  constructor(model, tokens, side) {
    this.model = model;
    this.tokens = tokens;
    this.side = side;
    this.canvas = new OffscreenCanvas(side, side);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.size = targetSize(side, side, tokens * 9);
  }
  chain = Promise.resolve();
  canvas;
  ctx;
  size;
  static async open({ weights = "weights", tokens = 70, side = 384, subgroupMatrix = true, onProgress } = {}) {
    const model = await load(weights, { subgroupMatrix, onProgress });
    return new _Eyes(model, tokens, side);
  }
  /** The GPU, as the browser names it. */
  get gpu() {
    const info = this.model.adapter.info || {};
    return info.description || (info.vendor ? `your ${info.vendor[0].toUpperCase() + info.vendor.slice(1)} GPU` : "your GPU");
  }
  /** A frame's unit embedding and the milliseconds EG2 took, one frame at a time (later calls wait their turn). */
  see(source) {
    const w = "videoWidth" in source ? source.videoWidth : source.width;
    const h = "videoHeight" in source ? source.videoHeight : source.height;
    const crop = Math.min(w, h), s = this.side;
    this.ctx.drawImage(source, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, s, s);
    const px = this.ctx.getImageData(0, 0, s, s).data, rgb = new Uint8Array(s * s * 3);
    for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
      rgb[j] = px[i];
      rgb[j + 1] = px[i + 1];
      rgb[j + 2] = px[i + 2];
    }
    const [th, tw] = this.size;
    const small = resize(rgb, s, s, th, tw);
    const run = this.chain.then(async () => {
      const t0 = performance.now();
      const x = await this.model.embedRgb(small, th, tw);
      return { x, ms: performance.now() - t0 };
    });
    this.chain = run.catch(() => {
    });
    return run;
  }
  /** The frame `see` last read, as a JPEG, for a teacher. */
  jpeg(quality = 0.85) {
    return this.canvas.convertToBlob({ type: "image/jpeg", quality });
  }
};
var TRAINER = `
onmessage = ({ data: { X, y, k, C, steps } }) => {
  const n = X.length, d = X[0].length;
  const mean = new Float64Array(d), sd = new Float64Array(d);
  for (const x of X) for (let j = 0; j < d; j++) mean[j] += x[j] / n;
  for (const x of X) for (let j = 0; j < d; j++) sd[j] += (x[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  const Z = X.map(x => Float64Array.from(x, (v, j) => (v - mean[j]) / sd[j]));
  const count = new Float64Array(k); for (const c of y) count[c]++;
  const cw = Array.from(count, c => c ? n / (k * c) : 0);
  const W = Array.from({ length: k }, () => new Float64Array(d)), b = new Float64Array(k);
  const mW = W.map(() => new Float64Array(d)), vW = W.map(() => new Float64Array(d)), mb = new Float64Array(k), vb = new Float64Array(k);
  for (let it = 1; it <= steps; it++) {
    const gW = W.map(() => new Float64Array(d)), gb = new Float64Array(k);
    for (let i = 0; i < n; i++) {
      const z = W.map((w, c) => { let s = b[c]; for (let j = 0; j < d; j++) s += w[j] * Z[i][j]; return s; });
      const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), t = e.reduce((a, v) => a + v, 0);
      for (let c = 0; c < k; c++) {
        const g = cw[y[i]] * (e[c] / t - (c === y[i] ? 1 : 0));
        gb[c] += g; const gw = gW[c], zi = Z[i];
        for (let j = 0; j < d; j++) gw[j] += g * zi[j];
      }
    }
    for (let c = 0; c < k; c++) {
      for (let j = 0; j < d; j++) {
        const g = C * gW[c][j] + W[c][j];
        mW[c][j] = 0.9 * mW[c][j] + 0.1 * g; vW[c][j] = 0.999 * vW[c][j] + 0.001 * g * g;
        W[c][j] -= 0.02 * (mW[c][j] / (1 - 0.9 ** it)) / (Math.sqrt(vW[c][j] / (1 - 0.999 ** it)) + 1e-8);
      }
      const g = C * gb[c];
      mb[c] = 0.9 * mb[c] + 0.1 * g; vb[c] = 0.999 * vb[c] + 0.001 * g * g;
      b[c] -= 0.02 * (mb[c] / (1 - 0.9 ** it)) / (Math.sqrt(vb[c] / (1 - 0.999 ** it)) + 1e-8);
    }
  }
  postMessage({ mean, sd, W, b });
};`;
function predict(fit, x) {
  return softmax2(fit.W.map((w, c) => {
    let s = fit.b[c];
    for (let j = 0; j < x.length; j++) s += w[j] * (x[j] - fit.mean[j]) / fit.sd[j];
    return s;
  }));
}
var Choice = class {
  names = [];
  prompts = [];
  labelled = [];
  start;
  quiet;
  empty;
  opts;
  fit = null;
  baseline = null;
  residuals = [];
  graduated = /* @__PURE__ */ new Set();
  checks = /* @__PURE__ */ new Map();
  checkedAt = /* @__PURE__ */ new Map();
  generation = 0;
  // forget() moves it on, so a fit begun before it isn't installed after it
  fitting = false;
  again = false;
  worker = new Worker(URL.createObjectURL(new Blob([TRAINER], { type: "text/javascript" })));
  /** `answers` maps each answer to its prompt vector (the mean of its prompts' unit embeddings), or null for one
   * that's only ever taught. */
  constructor(answers, opts = {}) {
    for (const [name, prompt] of Object.entries(answers)) this.add(name, prompt);
    this.start = this.names.length;
    this.quiet = new Set(opts.quiet ?? []);
    this.empty = opts.empty;
    this.opts = {
      temperature: 2.5,
      sure: 0.5,
      shown: 3,
      resemble: 0.6,
      resembleScale: 0.05,
      halfAt: 8,
      perAnswer: 60,
      target: 0.9,
      minChecks: 20,
      window: 100,
      gap: 1,
      ...Object.fromEntries(Object.entries(opts).filter(([k]) => k !== "quiet" && k !== "empty"))
    };
  }
  /** The answer a name means, allowing for a plural. */
  find(name) {
    for (const n of [name, name + "s", name.replace(/s+$/, ""), name.replace(/es$/, "")]) if (this.names.includes(n)) return n;
    return null;
  }
  /** An answer, added with its prompt vector (or none) unless there's one by that name already. */
  add(name, prompt = null) {
    const found = this.find(name);
    if (found !== null) return found;
    this.prompts.push(prompt ? Float32Array.from(prompt) : null);
    this.names.push(name);
    return name;
  }
  /** Labels frame `x` with `name` (added as an answer without prompts if it isn't one) and refits in the background;
   * returns the answer it was labelled with. With `check`, the label also checks the answer given for the frame
   * before it. */
  teach(x, name, by = "you", check = true) {
    const before = check ? this.ask(x) : null;
    const found = this.add(name);
    this.labelled.push({ x, i: this.names.indexOf(found), by });
    this.refit();
    if (before) this.check(before, found);
    return found;
  }
  /** Checks the answer `ask` gave for a frame against the truth for it (an answer, or null for none of them). Returns
   * whether they agreed, or null when the check doesn't count (not sure, or under `gap` seconds since the last). */
  check(answer, truth) {
    if (answer.state !== "sure") return null;
    const { name } = answer, now = performance.now() / 1e3, o = this.opts;
    if (now - (this.checkedAt.get(name) ?? -Infinity) < o.gap) return null;
    this.checkedAt.set(name, now);
    const record = [...this.checks.get(name) ?? [], truth === name].slice(-o.window);
    this.checks.set(name, record);
    const [low, high] = wilson(record.filter(Boolean).length, record.length);
    if (this.graduated.has(name)) {
      if (high < o.target) this.graduated.delete(name);
    } else if (record.length >= o.minChecks && low >= o.target) this.graduated.add(name);
    return truth === name;
  }
  /** Per checked answer, its `progress`. */
  report() {
    return Object.fromEntries([...this.checks.keys()].map((name) => [name, this.progress(name)]));
  }
  /** An answer's way to graduating: its checks, their agreement and its 95% bounds, whether it has graduated,
   * `needed`, the agreeing checks it still needs (0 once graduated; null if it can't within a window), and `done`, the
   * share of the way there: checks / (checks + needed). */
  progress(name) {
    const o = this.opts, record = this.checks.get(name) ?? [], graduated = this.graduated.has(name);
    const k = record.filter(Boolean).length, m = record.length, [low, high] = wilson(k, m);
    let needed = graduated ? 0 : null;
    if (!graduated) {
      let ahead = [...record];
      for (let more = 0; more <= o.window + o.minChecks; more++) {
        if (ahead.length >= o.minChecks && wilson(ahead.filter(Boolean).length, ahead.length)[0] >= o.target) {
          needed = more;
          break;
        }
        ahead = [...ahead, true].slice(-o.window);
      }
    }
    const done = graduated ? 1 : needed === null ? 0 : m / (m + needed);
    return { checks: m, agree: m ? k / m : null, low, high, graduated, needed, done };
  }
  /** Back to the answers it started with: the labelled frames go, and so do the answers added since. */
  forget() {
    this.generation++;
    this.labelled = [];
    this.fit = this.baseline = null;
    this.residuals = [];
    this.names = this.names.slice(0, this.start);
    this.prompts = this.prompts.slice(0, this.start);
    this.checks.clear();
    this.checkedAt.clear();
    this.graduated.clear();
  }
  /** Labelled frames per answer, by who labelled them. */
  counts() {
    const out = {};
    for (const { i, by } of this.labelled) {
      const c = out[this.names[i]] ??= {};
      c[by] = (c[by] ?? 0) + 1;
    }
    return out;
  }
  refit() {
    this.again = false;
    const e = this.empty === void 0 ? -1 : this.names.indexOf(this.empty);
    const E = this.labelled.filter((l) => l.i === e).map((l) => l.x);
    if (E.length >= 2) {
      const centre = new Float32Array(E[0].length);
      for (const x of E) for (let j = 0; j < x.length; j++) centre[j] += x[j] / E.length;
      const d = E.map((x) => norm2(sub(x, centre))).sort((a, b) => a - b);
      const mean = d.reduce((a, v) => a + v, 0) / d.length;
      const spread = Math.sqrt(d.reduce((a, v) => a + (v - mean) ** 2, 0) / d.length) + 1e-4;
      this.baseline = { centre, d: d[d.length >> 1], spread };
      this.residuals = this.labelled.filter((l) => !this.quiet.has(this.names[l.i])).map((l) => unit(sub(l.x, centre)));
    } else {
      this.baseline = null;
      this.residuals = [];
    }
    const classes = [...new Set(this.labelled.map((l) => l.i))].sort((a, b) => a - b);
    if (classes.length < 2) {
      this.fit = null;
      return;
    }
    if (this.fitting) {
      this.again = true;
      return;
    }
    this.fitting = true;
    const keep = classes.flatMap((c) => this.labelled.filter((l) => l.i === c).slice(-this.opts.perAnswer));
    const asked = this.generation;
    this.worker.onmessage = ({ data }) => {
      this.fitting = false;
      if (asked === this.generation) this.fit = { ...data, classes };
      if (this.again) this.refit();
    };
    this.worker.postMessage({ X: keep.map((l) => l.x), y: keep.map((l) => classes.indexOf(l.i)), k: classes.length, C: 0.5, steps: 150 });
  }
  /** The answer for frame `x`. */
  ask(x) {
    const { names, prompts, fit, baseline, residuals, opts } = this;
    const n = names.length, k = this.labelled.length;
    const prompted = prompts.map((p2) => p2 !== null);
    const s = prompts.flatMap((p2) => p2 ? [dot(p2, x)] : []);
    const mu = s.reduce((a, v) => a + v, 0) / s.length;
    const sd = Math.sqrt(s.reduce((a, v) => a + (v - mu) ** 2, 0) / s.length) + 1e-6;
    const zs = softmax2(s.map((v) => opts.temperature * (v - mu) / sd));
    let at = 0;
    const zero = prompted.map((p2) => p2 ? zs[at++] : 0);
    let p = zero, shown = null, resemble = 0, w = 0;
    if (baseline) {
      const r = sub(x, baseline.centre);
      shown = (norm2(r) - baseline.d) / baseline.spread;
      const ru = unit(r);
      for (const q of residuals) resemble = Math.max(resemble, dot(q, ru));
    }
    const student = fit && Math.max(...fit.classes) < n ? fit : null;
    if (student) {
      const taught = new Array(n).fill(0);
      predict(student, x).forEach((v, c) => taught[student.classes[c]] = v);
      if (!baseline) w = k / (k + opts.halfAt);
      else {
        w = residuals.length ? sigmoid((resemble - opts.resemble) / opts.resembleScale) : 0;
        w = Math.max(w, sigmoid(2 * (opts.shown - shown)));
      }
      p = zero.map((z, i) => (1 - w) * z + w * taught[i]);
    }
    let best = 0;
    for (let i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    const state = this.quiet.has(names[best]) ? "quiet" : p[best] >= opts.sure ? "sure" : shown === null || shown >= opts.shown ? "unsure" : "quiet";
    const mode = student ? `taught: ${k} labelled frames, student weight ${Math.round(w * 100)}%` : k ? `zero-shot until two answers have frames (${k} labelled)` : "zero-shot: text prompts only";
    const named = (v) => Object.fromEntries(names.map((name, i) => [name, v[i]]));
    return {
      name: names[best],
      p: p[best],
      state,
      graduated: this.graduated.has(names[best]),
      probs: named(p),
      zero: named(zero),
      shown,
      resemble,
      weight: w,
      ready: baseline !== null,
      mode
    };
  }
};
var Detector = class {
  /** Zero-shot, a frame's similarity to the `yes` prompt vector minus its similarity to the `no` one, against
   * `threshold`, with `scale` the distance past it that makes p = 0.73. Taught, the direction from the "no" frames to
   * the "yes" frames turns the prompts' direction by `nudge`, and the threshold moves halfway between them. */
  constructor(yes = null, no = null, threshold = 0, scale = 0.01, nudge = 0.5) {
    this.threshold = threshold;
    this.scale = scale;
    this.nudge = nudge;
    this.v0 = yes && no ? sub(yes, no) : null;
  }
  static names = ["no", "yes"];
  labelled = [];
  v0;
  taught = null;
  teach(x, yes) {
    this.labelled.push({ x, yes });
    this.refit();
  }
  forget() {
    this.labelled = [];
    this.taught = null;
  }
  counts() {
    const yes = this.labelled.filter((l) => l.yes).length;
    return { no: this.labelled.length - yes, yes };
  }
  refit() {
    const mean = (xs) => Float32Array.from(xs[0], (_, j) => xs.reduce((a2, x) => a2 + x[j], 0) / xs.length);
    const no = this.labelled.filter((l) => !l.yes).map((l) => l.x), yes = this.labelled.filter((l) => l.yes).map((l) => l.x);
    if (!no.length || !yes.length) return;
    const mn = mean(no), my = mean(yes), d = unit(sub(my, mn));
    const v = this.v0 ? Float32Array.from(unit(this.v0), (u, j) => u + this.nudge * d[j]) : d;
    const a = dot(mn, v), b = dot(my, v);
    this.taught = { v, cut: (a + b) / 2, scale: Math.max(Math.abs(b - a) / 8, 1e-4) };
  }
  /** p(yes), or null before it can answer; zero, the prompts' p(yes) or null. */
  ask(x) {
    const zero = this.v0 ? sigmoid((dot(x, this.v0) - this.threshold) / this.scale) : null;
    const t = this.taught;
    return { p: t ? sigmoid((dot(x, t.v) - t.cut) / t.scale) : zero, zero, taught: t !== null };
  }
};
var Watcher = class {
  constructor(choice, teacher, opts = {}) {
    this.choice = choice;
    this.teacher = teacher;
    this.opts = { by: "teacher", sure: 0.5, cap: 60, rest: 0.3, idle: 3, fresh: 1.5, auditEvery: 10, steady: 3, ...opts };
  }
  last = null;
  agree = [];
  looked = [];
  newest = null;
  streak = { name: null, n: 0 };
  auditAt = /* @__PURE__ */ new Map();
  looking = false;
  opts;
  /** Offers the newest frame and the student's answer for it; the teacher looks if that's worth it. */
  look(image, x, answer) {
    const now = performance.now() / 1e3, o = this.opts, name = answer.name;
    this.streak = answer.state === "sure" ? { name, n: this.streak.name === name ? this.streak.n + 1 : 1 } : { name: null, n: 0 };
    const steady = this.streak.n >= o.steady;
    let worth;
    if (answer.state === "quiet") {
      worth = (answer.shown ?? 0) >= this.choice.opts.shown && now - (this.looked.at(-1) ?? -Infinity) >= o.idle;
    } else if (answer.graduated) {
      worth = now >= (this.auditAt.get(name) ?? -Infinity);
      if (worth) this.auditAt.set(name, now - o.auditEvery * Math.log(1 - Math.random()));
    } else {
      worth = true;
    }
    if (!worth) return;
    this.newest = { image, x, answer, steady };
    if (!this.looking) this.watch();
  }
  forget() {
    this.agree = [];
    this.auditAt.clear();
    this.streak = { name: null, n: 0 };
    this.last = null;
  }
  async watch() {
    this.looking = true;
    try {
      while (this.newest) {
        const { image, x, answer, steady } = this.newest, o = this.opts;
        this.newest = null;
        this.looked = [...this.looked.slice(-199), performance.now() / 1e3];
        const t0 = performance.now();
        const out = await this.teacher(image, [...this.choice.names]);
        const ms = performance.now() - t0;
        const best = Object.keys(out).reduce((a, b) => out[b] > out[a] ? b : a);
        const name = best === ELSE ? null : this.choice.find(best);
        this.last = { name: name ?? best, known: name !== null, p: out[best], ms, at: performance.now() / 1e3 };
        if (out[best] >= o.sure && steady) {
          const agreed = this.choice.check(answer, name);
          if (agreed !== null) this.agree = [...this.agree.slice(-49), agreed];
        }
        if (out[best] >= o.sure && name !== null && (this.choice.counts()[name]?.[o.by] ?? 0) < o.cap) {
          this.choice.teach(x, name, o.by, false);
          o.onLabel?.(image, x, name);
        }
        await new Promise((r) => setTimeout(r, o.rest * 1e3));
      }
    } finally {
      this.looking = false;
    }
  }
  /** Whether the teacher's latest look, still fresh, was sure it saw none of the answers. */
  sawElse() {
    const l = this.last;
    return l !== null && !l.known && l.p >= this.opts.sure && performance.now() / 1e3 - l.at < this.opts.fresh;
  }
  describe() {
    const l = this.last;
    if (!l) return null;
    const now = performance.now() / 1e3;
    return {
      name: l.name,
      known: l.known,
      p: l.p,
      ms: Math.round(l.ms),
      age: now - l.at,
      fresh: now - l.at < this.opts.fresh,
      perMinute: this.looked.filter((t) => now - t < 60).length,
      agree: this.agree.length ? this.agree.filter(Boolean).length / this.agree.length : null,
      checked: this.agree.length,
      graduated: [...this.choice.graduated].sort()
    };
  }
};
export {
  Choice,
  Detector,
  ELSE,
  Eyes,
  Watcher,
  unit,
  wilson
};
