import { meanBodyScore, personPresent } from './tracker';

describe('personPresent', () => {
  test('thresholds the mean of the filled body+foot slots, ignoring face/hands', () => {
    const scores = new Array(133).fill(0.9);
    expect(personPresent(scores)).toBe(true);
    scores.fill(0.1, 0, 23);
    expect(personPresent(scores)).toBe(false); // high face/hand scores don't count
  });

  test('ignores the small-toe slots BlazePose never fills', () => {
    const scores = new Array(133).fill(0);
    scores.fill(0.5, 0, 23);
    scores[18] = 0; // left small toe
    scores[21] = 0; // right small toe
    expect(meanBodyScore(scores)).toBeCloseTo(0.5);
  });
});
