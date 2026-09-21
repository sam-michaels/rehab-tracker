// Same frozen fixtures as ml/tests/test_pipeline_fixtures.py (ADR 0005 C4).
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ExerciseDefinition } from '../types/definition';
import { measure } from './measure';

const SHARED = join(__dirname, '../../../shared');
const load = (...p: string[]) => JSON.parse(readFileSync(join(SHARED, ...p), 'utf8'));
const TOL_DEG = 1e-6;
const fixtures = readdirSync(join(SHARED, 'fixtures', 'pipeline')).filter((f) => f.endsWith('.json'));

const close = (a: number | null, b: number | null) =>
  a === null || b === null ? a === b : Math.abs(a - b) <= TOL_DEG;

test('fixtures exist', () => expect(fixtures.sort()).toEqual(['clean.json', 'edge.json', 'gaps.json']));

test.each(fixtures)('%s matches the Python reference', (file) => {
  const fx = load('fixtures', 'pipeline', file);
  const def: ExerciseDefinition = load('definitions', `${fx.definition}.json`);
  expect(def.version).toBe(fx.definition_version);
  const got = measure(def, fx.calibration, fx.input.t, fx.input.keypoints);
  const exp = fx.expected;

  expect(got.status).toEqual(exp.status);
  expect(got.out_of_plane).toEqual(exp.out_of_plane);
  const badAngle = got.angle.findIndex((a, i) => !close(a, exp.angle[i]));
  expect(badAngle === -1 ? null : { frame: badAngle, got: got.angle[badAngle], exp: exp.angle[badAngle] }).toBeNull();
  expect(got.reps.length).toBe(exp.reps.length);
  got.reps.forEach((r, i) => {
    const e = exp.reps[i];
    expect({ start: r.start, end: r.end, measurable: r.measurable }).toEqual({
      start: e.start, end: e.end, measurable: e.measurable,
    });
    for (const k of ['peak_fraction', 'angle_min', 'angle_max'] as const) {
      expect(close(r[k], e[k])).toBe(true);
    }
  });
});
