// Writes known arrays for ml/tests/test_npy_roundtrip.py (ADR 0008 C3).
// Usage: npx tsx scripts/write-npy-fixtures.ts <out-dir>
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { toNpy } from '../src/series/npy';

const out = process.argv[2];
if (!out) throw new Error('usage: write-npy-fixtures.ts <out-dir>');
mkdirSync(out, { recursive: true });

// Values are a pure function of the index so Python can rebuild them: v[i] = i * 0.25 - 3, with NaN/±Inf planted.
const seq = (n: number, T: typeof Float32Array | typeof Float64Array) => {
  const a = new T(n).map((_, i) => i * 0.25 - 3);
  if (n > 3) [a[1], a[2], a[3]] = [NaN, Infinity, -Infinity];
  return a;
};
const cases: [string, number[], typeof Float32Array | typeof Float64Array][] = [
  ['f4_1d', [7], Float32Array],
  ['f4_1xN', [1, 5], Float32Array],
  ['f4_TxKx3', [4, 5, 3], Float32Array],
  ['f8_1d', [9], Float64Array],
  ['f4_empty', [0, 5, 3], Float32Array],
];
for (const [name, shape, T] of cases) {
  writeFileSync(join(out, `${name}.npy`), toNpy(seq(shape.reduce((a, b) => a * b, 1), T), shape));
}
