// EG2's image path on WebGPU (model.js): the parts hunch.live uses.
export interface Model {
  adapter: GPUAdapter;
  embedRgb(rgb: Uint8Array, height: number, width: number): Promise<Float32Array>;
}
export function load(base?: string, opts?: { subgroupMatrix?: boolean; onProgress?: ((at: number, total: number) => void) | null }): Promise<Model>;
export function targetSize(h: number, w: number, maxPatches: number): [number, number];
export function resize(rgb: Uint8Array, h: number, w: number, th: number, tw: number): Uint8Array;
export function preprocess(blob: Blob, maxSoftTokens?: number): Promise<{ rgb: Uint8Array; height: number; width: number }>;
