// hunch.live in the browser: questions about a live stream of images (a webcam), answered from one EG2 embedding per
// frame computed on the GPU through WebGPU, and learned from the frames a person or a teacher labels as it runs.
// The same parts as the Python module (hunch/src/hunch/live.py), and the same rules:
//
//   Eyes      EG2's image path on WebGPU (eg2/model.js), one frame at a time. The browser runs no text path, so
//             prompts arrive as vectors embedded beforehand (eg2cpu's Model.embed, unit-normalised and averaged).
//   Choice    picks one answer: zero-shot from prompt vectors, and a student (logistic regression on standardised
//             embeddings, classes weighted equally, fitted in a web worker) that answers as far as a frame
//             resembles what it was taught, measured from the `empty` answer's frames. Answers can be added while it
//             runs, with or without a prompt vector, and forget() goes back to the ones it started with.
//   Detector  says yes or no: prompt vectors, plus a direction taught from one frame with the thing and one without.
//   Watcher   a teacher model watching the stream: it looks at the newest frame when that's worth it, one look at a
//             time, and labels the frames it's sure about for a Choice's student.
//
// There is no Album: frames stay in the tab.

import { load, resize, targetSize } from './eg2/model.js';

export const ELSE = 'something else'; // a teacher's answer for a frame that shows none of the answers

export type Vector = Float32Array;
export type State = 'quiet' | 'sure' | 'unsure';
export interface Answer {
  name: string;
  p: number;
  state: State;
  probs: Record<string, number>;
  zero: Record<string, number>; // 0 for an answer without prompts
  shown: number | null;
  resemble: number;
  weight: number;
  ready: boolean;
  mode: string;
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));
export const unit = (v: ArrayLike<number>): Vector => {
  const n = norm(v) || 1;
  return Float32Array.from(v, (x) => x / n);
};
const sub = (a: ArrayLike<number>, b: ArrayLike<number>) => Float32Array.from(a, (x, i) => x - b[i]);
const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
function softmax(z: number[]): number[] {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---- Eyes

export interface EyesOptions {
  weights?: string; // where manifest.json and weights.bin are (examples/browser/convert.py writes them)
  tokens?: number; // soft tokens per frame (70, 140, 280)
  side?: number; // frames are centre-cropped to a square this many pixels wide before the resize
  subgroupMatrix?: boolean;
  onProgress?: (bytes: number, total: number) => void;
}

export class Eyes {
  private chain: Promise<unknown> = Promise.resolve();
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  private size: [number, number];

  private constructor(readonly model: any, readonly tokens: number, readonly side: number) {
    this.canvas = new OffscreenCanvas(side, side);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
    this.size = targetSize(side, side, tokens * 9);
  }

  static async open({ weights = 'weights', tokens = 70, side = 384, subgroupMatrix = true, onProgress }: EyesOptions = {}) {
    const model = await load(weights, { subgroupMatrix, onProgress });
    return new Eyes(model, tokens, side);
  }

  /** The GPU, as the browser names it. */
  get gpu(): string {
    const info = this.model.adapter.info || {};
    return info.description || (info.vendor ? `your ${info.vendor[0].toUpperCase() + info.vendor.slice(1)} GPU` : 'your GPU');
  }

  /** A frame's unit embedding and the milliseconds EG2 took, one frame at a time (later calls wait their turn). */
  see(source: HTMLVideoElement | HTMLCanvasElement | ImageBitmap | OffscreenCanvas): Promise<{ x: Vector; ms: number }> {
    const w = 'videoWidth' in source ? source.videoWidth : source.width;
    const h = 'videoHeight' in source ? source.videoHeight : source.height;
    const crop = Math.min(w, h), s = this.side;
    this.ctx.drawImage(source as CanvasImageSource, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, s, s);
    const px = this.ctx.getImageData(0, 0, s, s).data, rgb = new Uint8Array(s * s * 3);
    for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
      rgb[j] = px[i]; rgb[j + 1] = px[i + 1]; rgb[j + 2] = px[i + 2];
    }
    const [th, tw] = this.size;
    const small = resize(rgb, s, s, th, tw);
    const run = this.chain.then(async () => {
      const t0 = performance.now();
      const x: Vector = await this.model.embedRgb(small, th, tw);
      return { x, ms: performance.now() - t0 };
    });
    this.chain = run.catch(() => {});
    return run;
  }

  /** The frame `see` last read, as a JPEG, for a teacher. */
  jpeg(quality = 0.85): Promise<Blob> {
    return this.canvas.convertToBlob({ type: 'image/jpeg', quality });
  }
}

// ---- the student: a logistic regression on standardised embeddings, classes weighted equally (sklearn's objective,
// C * loss + |W|^2 / 2, minimised with Adam), fitted in a worker so teaching at the frame rate never waits on it

interface Fit {
  mean: Float64Array;
  sd: Float64Array;
  W: Float64Array[];
  b: Float64Array;
  classes: number[];
}

const TRAINER = `
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

function predict(fit: Fit, x: Vector): number[] {
  return softmax(fit.W.map((w, c) => {
    let s = fit.b[c];
    for (let j = 0; j < x.length; j++) s += (w[j] * (x[j] - fit.mean[j])) / fit.sd[j];
    return s;
  }));
}

// ---- Choice

export interface ChoiceOptions {
  quiet?: Iterable<string>; // never named, never asked about
  empty?: string; // the answer whose frames are the baseline
  temperature?: number; // on standardised prompt similarities
  sure?: number; // an answer this likely is "sure"
  shown?: number; // a frame this many baseline spreads from the baseline frames is showing something
  resemble?: number; // likeness to the taught frames at which the student and the prompts weigh the same
  resembleScale?: number;
  halfAt?: number; // without a baseline, labelled frames at which the student and the prompts weigh the same
  perAnswer?: number; // the newest frames of each answer the student is fitted on
}

interface Labelled {
  x: Vector;
  i: number;
  by: string;
}

interface Baseline {
  centre: Vector;
  d: number;
  spread: number;
}

export class Choice {
  names: string[] = [];
  prompts: (Vector | null)[] = [];
  labelled: Labelled[] = [];
  readonly start: number;
  readonly quiet: Set<string>;
  readonly empty?: string;
  readonly opts: Required<Omit<ChoiceOptions, 'quiet' | 'empty'>>;
  private fit: Fit | null = null;
  private baseline: Baseline | null = null;
  private residuals: Vector[] = [];
  private generation = 0; // forget() moves it on, so a fit begun before it isn't installed after it
  private fitting = false;
  private again = false;
  private worker = new Worker(URL.createObjectURL(new Blob([TRAINER], { type: 'text/javascript' })));

  /** `answers` maps each answer to its prompt vector (the mean of its prompts' unit embeddings), or null for one
   * that's only ever taught. */
  constructor(answers: Record<string, ArrayLike<number> | null>, opts: ChoiceOptions = {}) {
    for (const [name, prompt] of Object.entries(answers)) this.add(name, prompt);
    this.start = this.names.length;
    this.quiet = new Set(opts.quiet ?? []);
    this.empty = opts.empty;
    this.opts = {
      temperature: 2.5, sure: 0.5, shown: 3.0, resemble: 0.6, resembleScale: 0.05, halfAt: 8, perAnswer: 60,
      ...Object.fromEntries(Object.entries(opts).filter(([k]) => k !== 'quiet' && k !== 'empty')),
    } as Choice['opts'];
  }

  /** The answer a name means, allowing for a plural. */
  find(name: string): string | null {
    for (const n of [name, name + 's', name.replace(/s+$/, ''), name.replace(/es$/, '')]) if (this.names.includes(n)) return n;
    return null;
  }

  /** An answer, added with its prompt vector (or none) unless there's one by that name already. */
  add(name: string, prompt: ArrayLike<number> | null = null): string {
    const found = this.find(name);
    if (found !== null) return found;
    this.prompts.push(prompt ? Float32Array.from(prompt) : null);
    this.names.push(name);
    return name;
  }

  /** Labels frame `x` with `name` (added as an answer without prompts if it isn't one) and refits in the background;
   * returns the answer it was labelled with. */
  teach(x: Vector, name: string, by = 'you'): string {
    const found = this.add(name);
    this.labelled.push({ x, i: this.names.indexOf(found), by });
    this.refit();
    return found;
  }

  /** Back to the answers it started with: the labelled frames go, and so do the answers added since. */
  forget(): void {
    this.generation++;
    this.labelled = [];
    this.fit = this.baseline = null;
    this.residuals = [];
    this.names = this.names.slice(0, this.start);
    this.prompts = this.prompts.slice(0, this.start);
  }

  /** Labelled frames per answer, by who labelled them. */
  counts(): Record<string, Record<string, number>> {
    const out: Record<string, Record<string, number>> = {};
    for (const { i, by } of this.labelled) {
      const c = (out[this.names[i]] ??= {});
      c[by] = (c[by] ?? 0) + 1;
    }
    return out;
  }

  private refit(): void {
    this.again = false;
    const e = this.empty === undefined ? -1 : this.names.indexOf(this.empty);
    const E = this.labelled.filter((l) => l.i === e).map((l) => l.x);
    if (E.length >= 2) { // the baseline frames: where they sit, and how far apart they spread
      const centre = new Float32Array(E[0].length);
      for (const x of E) for (let j = 0; j < x.length; j++) centre[j] += x[j] / E.length;
      const d = E.map((x) => norm(sub(x, centre))).sort((a, b) => a - b);
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
  ask(x: Vector): Answer {
    const { names, prompts, fit, baseline, residuals, opts } = this;
    const n = names.length, k = this.labelled.length;
    const prompted = prompts.map((p) => p !== null);
    const s = prompts.flatMap((p) => (p ? [dot(p, x)] : []));
    const mu = s.reduce((a, v) => a + v, 0) / s.length;
    const sd = Math.sqrt(s.reduce((a, v) => a + (v - mu) ** 2, 0) / s.length) + 1e-6;
    const zs = softmax(s.map((v) => (opts.temperature * (v - mu)) / sd));
    let at = 0;
    const zero = prompted.map((p) => (p ? zs[at++] : 0));
    let p = zero, shown: number | null = null, resemble = 0, w = 0;
    if (baseline) {
      const r = sub(x, baseline.centre);
      shown = (norm(r) - baseline.d) / baseline.spread;
      const ru = unit(r);
      for (const q of residuals) resemble = Math.max(resemble, dot(q, ru));
    }
    const student = fit && Math.max(...fit.classes) < n ? fit : null; // not one fitted before a forget
    if (student) {
      const taught = new Array(n).fill(0);
      predict(student, x).forEach((v, c) => (taught[student.classes[c]] = v));
      if (!baseline) w = k / (k + opts.halfAt);
      else { // as far as the frame looks like the things it was taught, or like the baseline
        w = residuals.length ? sigmoid((resemble - opts.resemble) / opts.resembleScale) : 0;
        w = Math.max(w, sigmoid(2 * (opts.shown - shown!)));
      }
      p = zero.map((z, i) => (1 - w) * z + w * taught[i]);
    }
    let best = 0;
    for (let i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    const state: State = this.quiet.has(names[best]) ? 'quiet'
      : p[best] >= opts.sure ? 'sure'
      : shown === null || shown >= opts.shown ? 'unsure' : 'quiet';
    const mode = student ? `taught: ${k} labelled frames, student weight ${Math.round(w * 100)}%`
      : k ? `zero-shot until two answers have frames (${k} labelled)` : 'zero-shot: text prompts only';
    const named = (v: number[]) => Object.fromEntries(names.map((name, i) => [name, v[i]]));
    return { name: names[best], p: p[best], state, probs: named(p), zero: named(zero), shown, resemble, weight: w,
             ready: baseline !== null, mode };
  }
}

// ---- Detector

export class Detector {
  static readonly names = ['no', 'yes'];
  labelled: { x: Vector; yes: boolean }[] = [];
  private v0: Vector | null;
  private taught: { v: Vector; cut: number; scale: number } | null = null;

  /** Zero-shot, a frame's similarity to the `yes` prompt vector minus its similarity to the `no` one, against
   * `threshold`, with `scale` the distance past it that makes p = 0.73. Taught, the direction from the "no" frames to
   * the "yes" frames turns the prompts' direction by `nudge`, and the threshold moves halfway between them. */
  constructor(yes: ArrayLike<number> | null = null, no: ArrayLike<number> | null = null,
              readonly threshold = 0, readonly scale = 0.01, readonly nudge = 0.5) {
    this.v0 = yes && no ? sub(yes, no) : null;
  }

  teach(x: Vector, yes: boolean): void {
    this.labelled.push({ x, yes });
    this.refit();
  }

  forget(): void {
    this.labelled = [];
    this.taught = null;
  }

  counts() {
    const yes = this.labelled.filter((l) => l.yes).length;
    return { no: this.labelled.length - yes, yes };
  }

  private refit(): void {
    const mean = (xs: Vector[]) => Float32Array.from(xs[0], (_, j) => xs.reduce((a, x) => a + x[j], 0) / xs.length);
    const no = this.labelled.filter((l) => !l.yes).map((l) => l.x), yes = this.labelled.filter((l) => l.yes).map((l) => l.x);
    if (!no.length || !yes.length) return;
    const mn = mean(no), my = mean(yes), d = unit(sub(my, mn));
    const v = this.v0 ? Float32Array.from(unit(this.v0), (u, j) => u + this.nudge * d[j]) : d;
    const a = dot(mn, v), b = dot(my, v);
    this.taught = { v, cut: (a + b) / 2, scale: Math.max(Math.abs(b - a) / 8, 1e-4) }; // taught frames at p ~ 0.02, 0.98
  }

  /** p(yes), or null before it can answer; zero, the prompts' p(yes) or null. */
  ask(x: Vector): { p: number | null; zero: number | null; taught: boolean } {
    const zero = this.v0 ? sigmoid((dot(x, this.v0) - this.threshold) / this.scale) : null;
    const t = this.taught;
    return { p: t ? sigmoid((dot(x, t.v) - t.cut) / t.scale) : zero, zero, taught: t !== null };
  }
}

// ---- Watcher

export type Teacher = (image: Blob, names: string[]) => Promise<Record<string, number>>;

export interface WatcherOptions {
  by?: string;
  sure?: number;
  cap?: number;
  rest?: number; // seconds
  confirmed?: number;
  idle?: number;
  fresh?: number;
  onLabel?: (image: Blob, x: Vector, name: string) => void;
}

/** A teacher model watching the stream for a Choice. It looks at the newest frame when the student's answer is a
 * thing it hasn't agreed on in the last `confirmed` seconds, or, when the answer is quiet, every `idle` seconds while
 * something is being shown, resting `rest` seconds between looks. Sure of an answer, it labels the frame for the
 * student, up to `cap` frames per answer. */
export class Watcher {
  last: { name: string; known: boolean; p: number; ms: number; at: number } | null = null;
  agree: boolean[] = [];
  private agreed = new Map<string, number>();
  private looked: number[] = [];
  private newest: { image: Blob; x: Vector } | null = null;
  private looking = false;
  readonly opts: Required<Omit<WatcherOptions, 'onLabel'>> & Pick<WatcherOptions, 'onLabel'>;

  constructor(readonly choice: Choice, readonly teacher: Teacher, opts: WatcherOptions = {}) {
    this.opts = { by: 'teacher', sure: 0.5, cap: 60, rest: 0.3, confirmed: 3, idle: 3, fresh: 1.5, ...opts };
  }

  /** Offers the newest frame and the student's answer for it; the teacher looks if that's worth it. */
  look(image: Blob, x: Vector, answer: Answer): void {
    const now = performance.now() / 1000, o = this.opts;
    const worth = answer.state === 'quiet'
      ? (answer.shown ?? 0) >= this.choice.opts.shown && now - (this.looked.at(-1) ?? -Infinity) >= o.idle
      : now - (this.agreed.get(answer.name) ?? -Infinity) > o.confirmed;
    if (!worth) return;
    this.newest = { image, x };
    if (!this.looking) this.watch();
  }

  forget(): void {
    this.agree = [];
    this.agreed.clear();
    this.last = null;
  }

  private async watch() {
    this.looking = true;
    try {
      while (this.newest) {
        const { image, x } = this.newest, o = this.opts;
        this.newest = null;
        this.looked = [...this.looked.slice(-199), performance.now() / 1000];
        const t0 = performance.now();
        const out = await this.teacher(image, [...this.choice.names]);
        const ms = performance.now() - t0;
        const best = Object.keys(out).reduce((a, b) => (out[b] > out[a] ? b : a));
        const name = best === ELSE ? null : this.choice.find(best);
        this.last = { name: name ?? best, known: name !== null, p: out[best], ms, at: performance.now() / 1000 };
        if (out[best] >= o.sure && name !== null) {
          const agreed = this.choice.ask(x).name === name;
          this.agree = [...this.agree.slice(-49), agreed];
          if (agreed) this.agreed.set(name, performance.now() / 1000);
          if ((this.choice.counts()[name]?.[o.by] ?? 0) < o.cap) {
            this.choice.teach(x, name, o.by);
            o.onLabel?.(image, x, name);
          }
        }
        await new Promise((r) => setTimeout(r, o.rest * 1000));
      }
    } finally {
      this.looking = false;
    }
  }

  /** Whether the teacher's latest look, still fresh, was sure it saw none of the answers. */
  sawElse(): boolean {
    const l = this.last;
    return l !== null && !l.known && l.p >= this.opts.sure && performance.now() / 1000 - l.at < this.opts.fresh;
  }

  describe() {
    const l = this.last;
    if (!l) return null;
    const now = performance.now() / 1000;
    return { name: l.name, known: l.known, p: l.p, ms: Math.round(l.ms), age: now - l.at, fresh: now - l.at < this.opts.fresh,
             perMinute: this.looked.filter((t) => now - t < 60).length,
             agree: this.agree.length ? this.agree.filter(Boolean).length / this.agree.length : null, checked: this.agree.length };
  }
}
