import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

/**
 * Result of one `pose()` call. All coordinates are normalized [0,1] in the
 * Frame buffer's coordinate space (origin top-left).
 */
export interface PoseResult {
  /** x0,y0,x1,y1,...,x132,y132 — 133 keypoints, flattened. */
  keypoints: number[];
  /** Per-keypoint confidence, 133 entries. */
  scores: number[];
  /** Detector box [x,y,w,h], only present when the detector ran. */
  detBox?: number[];
  detScore?: number;
  detMs?: number;
  poseMs: number;
}

export interface Pose extends HybridObject<{ ios: 'swift' }> {
  /**
   * Runs pose inference on `frame`.
   * `roi`: [x,y,w,h] normalized [0,1], or omitted to run the detector on the
   * whole frame first and derive the ROI from its box.
   */
  run(frame: Frame, roi?: number[]): PoseResult;
}
