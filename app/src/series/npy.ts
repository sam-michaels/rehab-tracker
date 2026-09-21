// Minimal .npy v1.0 writer: C order, little-endian. Python reads it with np.load (ADR 0008 8c).
// Spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

export function toNpy(data: Float32Array | Float64Array, shape: number[]): Uint8Array {
  const count = shape.reduce((a, b) => a * b, 1);
  if (count !== data.length) throw new Error(`shape ${shape} needs ${count} values, got ${data.length}`);

  const f64 = data instanceof Float64Array;
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(', ');
  let header = `{'descr': '<f${f64 ? 8 : 4}', 'fortran_order': False, 'shape': (${dims}), }`;
  // magic(6) + version(2) + header_len(2) + header + '\n' must be a multiple of 64.
  header += ' '.repeat(63 - ((10 + header.length) % 64)) + '\n';

  const out = new Uint8Array(10 + header.length + data.length * data.BYTES_PER_ELEMENT);
  const view = new DataView(out.buffer);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]); // \x93NUMPY v1.0
  view.setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);

  // DataView, not the typed array's buffer, so byte order never depends on the host.
  let off = 10 + header.length;
  for (const v of data) {
    if (f64) view.setFloat64(off, v, true);
    else view.setFloat32(off, v, true);
    off += data.BYTES_PER_ELEMENT;
  }
  return out;
}
