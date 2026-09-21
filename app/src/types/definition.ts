// Hand-written mirrors of shared/schemas/*. No codegen (ADR 0008 C2): src/__tests__/schemas.test.ts
// keeps these honest by validating the shared JSON files and type-checking them against these types.

/** COCO-WholeBody keypoint index, 0–132. */
export type Keypoint = number;
export type Segment = [from: Keypoint, to: Keypoint];

export interface Rule {
  id: string;
  signal: string;
  comparator: '<' | '<=' | '>' | '>=';
  threshold: number;
  citation: string;
  message: string;
}

export interface ExerciseDefinition {
  id: string;
  version: number;
  name: string;
  keypoints: Keypoint[];
  measured_angle: { segment_a: Segment; segment_b: Segment };
  counting_signal: { kind: 'keypoint_y'; keypoint: Keypoint };
  filter: {
    median_window: number;
    one_euro: { min_cutoff: number; beta: number; d_cutoff: number };
    min_confidence: number;
    max_gap_s: number;
  };
  rep_thresholds: { enter: number; exit: number };
  out_of_plane: { foot: Segment; shank: Segment; tolerance: number };
  rules: Rule[];
}

export interface KeypointSeriesSidecar {
  format_version: 1;
  keypoints_file: string;
  timestamps_file: string;
  keypoint_indices: Keypoint[];
  image_size: [width: number, height: number];
  model_id: string;
  source: { kind: 'device' | 'runner'; name: string };
}
