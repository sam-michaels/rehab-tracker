// Inference spike screen (ADR 0002 C1): camera -> pose plugin -> tracker, with a leg/foot
// overlay and the numbers the spike report needs. Throwaway by charter.
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import { Camera, useCameraPermission, useFrameOutput, type Frame } from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';
import { pose } from './modules/pose/src';
import { meanBodyScore, nextRoi, type Mode, type Roi } from './src/pose/tracker';
import { createStats, pushSample, summary, type Summary } from './src/pose/stats';
import { OneEuro } from './src/pipeline/measure';

// COCO-WholeBody: 11-16 hips/knees/ankles, 17-22 feet (big toe, small toe, heel per side).
const LEG = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
const FOOT = [17, 18, 19, 20, 21, 22];
// Overlay smoothing (the pipeline's One Euro, per coordinate). Tuning knob: these are for
// normalized 0-1 coords at ~10 Hz, not the definition's degrees. Lower minCutoff = steadier
// at rest; higher beta = less lag when moving.
const EURO = { minCutoff: 0.7, beta: 4, dCutoff: 1 };

interface View_ {
  ts: number; // seconds, for the filter
  points: number[]; // x,y pairs for LEG, normalized upright-image coords
  scores: number[];
  footConf: number;
  bodyConf: number; // mean body+foot score; < 0.3 -> model output is noise (it always emits keypoints), hide dots
  detScore: number; // last detector confidence
  box: number[] | null; // current pose crop [x,y,w,h] (or detector box), display coords
  aspect: number; // upright width / height
  stats: Summary;
}

// Worklet-runtime state lives on globalThis: closures captured by worklets are copies.
declare const globalThis: { __spike?: { roi: Roi | null; stats: ReturnType<typeof createStats>; n: number; det: number; box: number[] | null } };

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [mode, setMode] = useState<Mode>('B');
  const [facing, setFacing] = useState<'back' | 'front'>('back');
  const [view, setView] = useState<View_ | null>(null);
  // Counted rather than console.warn'd per drop (vision-camera's default), which floods the JS thread in dev.
  const dropped = useRef(0);
  const euro = useRef<OneEuro[] | null>(null);
  const showView = useCallback((v: View_) => {
    if (v.bodyConf < 0.3) euro.current = null; // person gone: restart rather than glide from stale spots
    else {
      const f = (euro.current ??= v.points.map(() => new OneEuro(EURO.minCutoff, EURO.beta, EURO.dCutoff)));
      v.points = v.points.map((p, k) => f[k].filter(p, v.ts));
    }
    setView(v);
  }, []);

  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet';
      const s = (globalThis.__spike ??= { roi: null, stats: createStats(), n: 0, det: 0, box: null });
      try {
        const redetect = s.roi === null;
        const r = pose(frame, s.roi);
        // Vision applies the frame orientation, so outputs are in the upright image.
        const sideways = frame.orientation === 'left' || frame.orientation === 'right';
        const aspect = sideways ? frame.height / frame.width : frame.width / frame.height;
        s.roi = nextRoi(mode, r, aspect);
        if (r.detScore !== undefined) s.det = r.detScore;
        // Draw the crop the pose model tracks with (current), falling back to the detector box.
        const b = s.roi ?? r.detBox ?? null;
        s.box = b && frame.isMirrored ? [1 - b[0] - b[2], b[1], b[2], b[3]] : b;
        pushSample(s.stats, { ts: Date.now() / 1000, poseMs: r.poseMs, detMs: r.detMs, redetected: redetect });
        if (++s.n % 3 === 0) {
          const points: number[] = [];
          const scores: number[] = [];
          let foot = 0;
          // Front camera: the preview is mirrored but the buffer (and so the keypoints) isn't.
          const mx = frame.isMirrored;
          for (const i of LEG) {
            const x = r.keypoints[2 * i];
            points.push(mx ? 1 - x : x, r.keypoints[2 * i + 1]), scores.push(r.scores[i]);
          }
          for (const i of FOOT) foot += r.scores[i];
          scheduleOnRN(showView, { ts: Date.now() / 1000, points, scores, footConf: foot / FOOT.length, bodyConf: meanBodyScore(r.scores), detScore: s.det, box: s.box, aspect, stats: summary(s.stats) });
        }
      } finally {
        frame.dispose();
      }
    },
    [mode, showView],
  );
  const frameOutput = useFrameOutput({ onFrame, onFrameDropped: () => void dropped.current++ });

  // Auto-log every 2 s so a test can run with the phone propped out of reach (read it in Metro).
  const log = () =>
    console.log('[spike]', JSON.stringify({ mode, facing, ...view?.stats, footConf: view?.footConf, bodyConf: view?.bodyConf, detScore: view?.detScore, dropped: dropped.current, box: view?.box?.map((v) => +v.toFixed(3)), points: view?.points.map((v) => +v.toFixed(3)) }));
  const logRef = useRef(log);
  logRef.current = log;
  useEffect(() => {
    const id = setInterval(() => logRef.current(), 2000);
    return () => clearInterval(id);
  }, []);

  if (!hasPermission) return <View style={styles.center}><Button title="Allow camera" onPress={requestPermission} /></View>;

  const st = view?.stats;
  return (
    <View style={styles.root}>
      <View style={{ width: '100%', aspectRatio: view?.aspect ?? 9 / 16 }}>
        <Camera style={StyleSheet.absoluteFill} device={facing} isActive outputs={[frameOutput]} resizeMode="contain" />
        {view?.box && (
          <View
            style={[
              styles.box,
              {
                left: `${view.box[0] * 100}%`,
                top: `${view.box[1] * 100}%`,
                width: `${view.box[2] * 100}%`,
                height: `${view.box[3] * 100}%`,
              },
            ]}
          />
        )}
        {view && view.bodyConf >= 0.3 &&
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
          total {st?.totalMsP50.toFixed(1)}/{st?.totalMsP95.toFixed(1)} ms · foot conf {view?.footConf.toFixed(2)} · dropped {dropped.current}{'\n'}
          det score {view?.detScore.toFixed(2)} · body conf {view?.bodyConf.toFixed(2)}
        </Text>
        <View style={styles.row}>
          <Button title={`Switch to ${mode === 'A' ? 'B' : 'A'}`} onPress={() => setMode(mode === 'A' ? 'B' : 'A')} />
          <Button title="Flip camera" onPress={() => setFacing(facing === 'back' ? 'front' : 'back')} />
          <Button title="Log" onPress={log} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000', paddingTop: 50 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  box: { position: 'absolute', borderWidth: 2, borderColor: '#ff0' },
  // Big enough to read from where you stand during a test (2-3 m from the phone).
  dot: { position: 'absolute', width: 20, height: 20, marginLeft: -10, marginTop: -10, borderRadius: 10, borderWidth: 2, borderColor: '#fff' },
  panel: { padding: 12 },
  text: { color: '#fff', fontFamily: 'Menlo', fontSize: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 8 },
});
