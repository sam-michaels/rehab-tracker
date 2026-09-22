// Inference spike screen (ADR 0002 C1): camera -> pose plugin -> tracker, with a leg/foot
// overlay and the numbers the spike report needs. Throwaway by charter.
import { useCallback, useRef, useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import { Camera, useCameraPermission, useFrameOutput, type Frame } from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';
import { pose } from './modules/pose/src';
import { nextRoi, personPresent, type Mode, type Roi } from './src/pose/tracker';
import { createStats, pushSample, summary, type Summary } from './src/pose/stats';

// COCO-WholeBody: 11-16 hips/knees/ankles, 17-22 feet (big toe, small toe, heel per side).
const LEG = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
const FOOT = [17, 18, 19, 20, 21, 22];

interface View_ {
  points: number[]; // x,y pairs for LEG, normalized upright-image coords
  scores: number[];
  footConf: number;
  person: boolean; // false -> model output is noise (it always emits keypoints); hide the dots
  aspect: number; // upright width / height
  stats: Summary;
}

// Worklet-runtime state lives on globalThis: closures captured by worklets are copies.
declare const globalThis: { __spike?: { roi: Roi | null; stats: ReturnType<typeof createStats>; n: number } };

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [mode, setMode] = useState<Mode>('B');
  const [view, setView] = useState<View_ | null>(null);
  // Counted rather than console.warn'd per drop (vision-camera's default), which floods the JS thread in dev.
  const dropped = useRef(0);

  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet';
      const s = (globalThis.__spike ??= { roi: null, stats: createStats(), n: 0 });
      try {
        const redetect = s.roi === null;
        const r = pose(frame, s.roi);
        // Vision applies the frame orientation, so outputs are in the upright image.
        const sideways = frame.orientation === 'left' || frame.orientation === 'right';
        const aspect = sideways ? frame.height / frame.width : frame.width / frame.height;
        s.roi = nextRoi(mode, r, aspect);
        pushSample(s.stats, { ts: Date.now() / 1000, poseMs: r.poseMs, detMs: r.detMs, redetected: redetect });
        if (++s.n % 3 === 0) {
          const points: number[] = [];
          const scores: number[] = [];
          let foot = 0;
          for (const i of LEG) points.push(r.keypoints[2 * i], r.keypoints[2 * i + 1]), scores.push(r.scores[i]);
          for (const i of FOOT) foot += r.scores[i];
          scheduleOnRN(setView, { points, scores, footConf: foot / FOOT.length, person: personPresent(r.scores), aspect, stats: summary(s.stats) });
        }
      } finally {
        frame.dispose();
      }
    },
    [mode],
  );
  const frameOutput = useFrameOutput({ onFrame, onFrameDropped: () => void dropped.current++ });

  if (!hasPermission) return <View style={styles.center}><Button title="Allow camera" onPress={requestPermission} /></View>;

  const st = view?.stats;
  return (
    <View style={styles.root}>
      <View style={{ width: '100%', aspectRatio: view?.aspect ?? 9 / 16 }}>
        <Camera style={StyleSheet.absoluteFill} device="back" isActive outputs={[frameOutput]} resizeMode="contain" />
        {view?.person &&
          LEG.map((_, j) => (
            <View
              key={j}
              style={[
                styles.dot,
                {
                  left: `${view.points[2 * j] * 100}%`,
                  top: `${view.points[2 * j + 1] * 100}%`,
                  backgroundColor: view.scores[j] >= 0.3 ? '#0f0' : '#f00',
                },
              ]}
            />
          ))}
      </View>
      <View style={styles.panel}>
        <Text style={styles.text}>
          mode {mode} · {st ? st.fps.toFixed(1) : '-'} fps · redetect {st ? (st.redetectRate * 100).toFixed(0) : '-'}%{'\n'}
          det p50/p95 {st?.detMsP50.toFixed(1)}/{st?.detMsP95.toFixed(1)} ms · pose {st?.poseMsP50.toFixed(1)}/
          {st?.poseMsP95.toFixed(1)} ms{'\n'}
          total {st?.totalMsP50.toFixed(1)}/{st?.totalMsP95.toFixed(1)} ms · foot conf {view?.footConf.toFixed(2)} · dropped {dropped.current}
        </Text>
        <View style={styles.row}>
          <Button title={`Switch to ${mode === 'A' ? 'B' : 'A'}`} onPress={() => setMode(mode === 'A' ? 'B' : 'A')} />
          <Button title="Log" onPress={() => console.log('[spike]', JSON.stringify({ mode, ...view?.stats, footConf: view?.footConf, dropped: dropped.current }))} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000', paddingTop: 50 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  dot: { position: 'absolute', width: 8, height: 8, marginLeft: -4, marginTop: -4, borderRadius: 4 },
  panel: { padding: 12 },
  text: { color: '#fff', fontFamily: 'Menlo', fontSize: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 8 },
});
