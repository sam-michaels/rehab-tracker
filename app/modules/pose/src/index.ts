import { NitroModules } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';
import type { Pose as PoseHybridObject, PoseResult } from './Pose.nitro';

export type { PoseResult };

const poseObject = NitroModules.createHybridObject<PoseHybridObject>('Pose');

/**
 * Runs pose inference on `frame`, worklet-callable from a vision-camera
 * frame processor. Tracking is native; `forceDetect` re-runs the person
 * detector on this frame regardless.
 */
export function pose(frame: Frame, forceDetect = false): PoseResult {
  'worklet';
  return poseObject.run(frame, forceDetect);
}
