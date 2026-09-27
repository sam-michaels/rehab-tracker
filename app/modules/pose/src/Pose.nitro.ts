import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

/**
 * Result of one `pose()` call. All coordinates are normalized [0,1] in the
 * Frame buffer's coordinate space (origin top-left).
 */
export interface PoseResult {
  /**
   * x0,y0,x1,y1,...,x132,y132: 133 COCO-WholeBody slots, flattened. BlazePose's landmarks
   * fill the body and foot slots (heel, and foot index as big toe); slots it has no
   * equivalent for (small toes 18/21, face, hands) are 0 with score 0.
   */
  keypoints: number[];
  /** Per-keypoint confidence (BlazePose visibility), 133 entries. All 0 when nobody is in view. */
  scores: number[];
  /** Detector box [x,y,w,h], only present when the detector ran and found someone. */
  detBox?: number[];
  detScore?: number;
  detMs?: number;
  poseMs: number;
}

export interface Pose extends HybridObject<{ ios: 'swift' }> {
  /**
   * Runs pose inference on `frame`. The plugin tracks natively: the detector runs on the
   * first frame and whenever the pose is lost, otherwise the ROI follows the previous
   * frame's landmarks (ADR 0002 C2). `forceDetect` runs the detector anyway (spike mode A).
   */
  run(frame: Frame, forceDetect?: boolean): PoseResult;
}
