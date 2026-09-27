// Person-presence check on the plugin's output. ROI tracking itself (ADR 0002 Condition 2:
// amortize the detector) is native since the switch to BlazePose: the plugin derives each
// frame's ROI from the previous frame's landmarks, as MediaPipe does. Mode A forces the
// detector every frame so the saving stays measurable.
// Pure, worklet-safe: plain data in and out, no closures over module state.

export type Mode = 'A' | 'B';

// COCO-WholeBody body+foot slots BlazePose fills (0-17, 19, 20, 22). The small toes (18, 21)
// have no BlazePose equivalent and are always 0, so they would drag the mean down.
const BODY_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22];
// Reuse the definition's filter.min_confidence concept (shared/definitions/calf-raise.json).
const DEFAULT_MIN_CONFIDENCE = 0.3;

/** Mean score of the body+foot slots BlazePose fills. */
export function meanBodyScore(scores: number[]): number {
  'worklet';
  let sum = 0;
  for (const i of BODY_SLOTS) sum += scores[i];
  return sum / BODY_SLOTS.length;
}

/** True if a person is plausibly in view: meanBodyScore >= threshold. */
export function personPresent(scores: number[], threshold?: number): boolean {
  'worklet';
  threshold ??= DEFAULT_MIN_CONFIDENCE;
  return meanBodyScore(scores) >= threshold;
}
