// Same files, same schemas as ml/tests/test_schemas.py (ADR 0008 C2).
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import Ajv2020 from 'ajv/dist/2020';
import type { ExerciseDefinition, KeypointSeriesSidecar } from '../types/definition';

const SHARED = join(__dirname, '../../../shared');
const load = (...p: string[]) => JSON.parse(readFileSync(join(SHARED, ...p), 'utf8'));
// Formats (e.g. date) are checked on the Python side; ajv needs a plugin for them.
const ajv = new Ajv2020({ allErrors: true, validateFormats: false });
const validate = (name: string) =>
  ajv.getSchema(`${name}.schema.json`) ?? ajv.compile(load('schemas', `${name}.schema.json`));

const definitions = readdirSync(join(SHARED, 'definitions')).filter((f) => f.endsWith('.json'));

test('there are definitions', () => expect(definitions.length).toBeGreaterThan(0));

test.each(definitions)('%s is valid', (file) => {
  const v = validate('exercise-definition');
  const def: ExerciseDefinition = load('definitions', file);
  expect(v(def) ? null : v.errors).toBeNull();
});

test('rule without citation is rejected', () => {
  const def: ExerciseDefinition = load('definitions', 'calf-raise.json');
  const rule = { id: 'r', signal: 's', comparator: '<', threshold: 1, message: 'm' };
  expect(validate('exercise-definition')({ ...def, rules: [rule] })).toBe(false);
});

test('example sidecar is valid', () => {
  const sidecar: KeypointSeriesSidecar = load('fixtures', 'examples', 'keypoint-series.json');
  const v = validate('keypoint-series');
  expect(v(sidecar) ? null : v.errors).toBeNull();
});

test('example clip manifest is valid', () => {
  const v = validate('clip-manifest');
  expect(v(load('fixtures', 'examples', 'clip-manifest.json')) ? null : v.errors).toBeNull();
});
