import { NitroModules } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';
import type { Pose as PoseHybridObject, PoseResult } from './Pose.nitro';

export type { PoseResult };

const poseObject = NitroModules.createHybridObject<PoseHybridObject>('Pose');

/**
 * Runs pose inference on `frame`, worklet-callable from a vision-camera
 * frame processor.
 *
 * `roi` — [x,y,w,h] normalized [0,1] in the frame buffer's coordinate space
 * (origin top-left), or `null` to run the person detector on the whole
 * frame first and derive the ROI from it.
 */
export function pose(frame: Frame, roi: number[] | null): PoseResult {
  'worklet';
  return poseObject.run(frame, roi ?? undefined);
}
