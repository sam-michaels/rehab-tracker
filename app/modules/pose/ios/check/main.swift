// Asserts BlazePoseGeometry.swift against shared/fixtures/blazepose/, the same files
// ml/tests/test_blazepose_fixtures.py checks the Python reference against. Not part of the pod
// (the podspec globs ios/*.swift only). Run from the repo root:
//
//   swiftc -O app/modules/pose/ios/BlazePoseGeometry.swift app/modules/pose/ios/check/main.swift -o /tmp/bp-check
//   /tmp/bp-check shared/fixtures/blazepose
import Foundation

let tol = 1e-7
let heatmapFill: Float = 0  // as HEATMAP_FILL in ml/runners/make_blazepose_fixtures.py
var failures = 0

func check(_ name: String, _ got: [Double], _ exp: [Double]) {
  guard got.count == exp.count else {
    print("  FAIL \(name): count \(got.count) != \(exp.count)")
    failures += 1
    return
  }
  let worst = zip(got, exp).map { abs($0 - $1) }.max() ?? 0
  if worst > tol {
    print("  FAIL \(name): max abs diff \(worst)")
    failures += 1
  }
}

func doubles(_ v: Any?) -> [Double] { (v as! [Any]).map { ($0 as! NSNumber).doubleValue } }
func floats(_ v: Any?) -> [Float] { doubles(v).map(Float.init) }
func flat(_ pts: [(Double, Double)]) -> [Double] { pts.flatMap { [$0.0, $0.1] } }
func rectArray(_ r: PoseRect) -> [Double] { [r.cx, r.cy, r.w, r.h, r.rotation] }

let dir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "shared/fixtures/blazepose")
let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
  .filter { $0.pathExtension == "json" }.sorted { $0.path < $1.path }
guard files.count == 3 else { fatalError("expected 3 fixtures in \(dir.path), found \(files.count)") }
check("anchors", [Double(BlazePose.anchors.count)], [2254])

for file in files {
  print(file.lastPathComponent)
  let fx = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
  let size = doubles(fx["image_size"])
  let (w, h) = (size[0], size[1])
  let det = fx["detector"] as! [String: Any]
  let lm = fx["landmarks"] as! [String: Any]
  let exp = fx["expected"] as! [String: Any]

  var boxes = [Float](repeating: 0, count: BlazePose.anchors.count * 12)
  for (i, row) in det["boxes"] as! [String: Any] {
    for (k, v) in floats(row).enumerated() { boxes[Int(i)! * 12 + k] = v }
  }
  guard let d = BlazePose.bestDetection(rawBoxes: boxes, rawScores: floats(det["scores"])) else {
    fatalError("no detection")
  }
  let ed = exp["detection"] as! [String: Any]
  check("detection.score", [d.score], [(ed["score"] as! NSNumber).doubleValue])
  check("detection.box", d.box, doubles(ed["box"]))
  check("detection.keypoints", flat(d.keypoints), (ed["keypoints"] as! [Any]).flatMap { doubles($0) })

  let rect = BlazePose.rectFromDetection(d, width: w, height: h)
  check("rect", rectArray(rect), doubles(exp["rect"]))

  var heatmap = [Float](repeating: heatmapFill, count: 64 * 64 * BlazePose.numLandmarks)
  for (ch, win) in (lm["windows"] as! [[Any]]).enumerated() {
    let r0 = (win[0] as! NSNumber).intValue, c0 = (win[1] as! NSNumber).intValue
    for (dr, row) in (win[2] as! [Any]).enumerated() {
      for (dc, v) in floats(row).enumerated() {
        heatmap[((r0 + dr) * 64 + c0 + dc) * BlazePose.numLandmarks + ch] = v
      }
    }
  }
  let l = BlazePose.decodeLandmarks(
    raw: floats(lm["raw"]), flag: floats(lm["flag"])[0], heatmap: heatmap, rect: rect)
  check("presence", [l.presence], [(exp["presence"] as! NSNumber).doubleValue])
  check("points", flat(l.points), (exp["points"] as! [Any]).flatMap { doubles($0) })
  check("visibility", l.visibility, doubles(exp["visibility"]))
  let (kps, scores) = BlazePose.toWholeBody(l)
  check("wholebody_keypoints", kps, doubles(exp["wholebody_keypoints"]))
  check("wholebody_scores", scores, doubles(exp["wholebody_scores"]))
  check("next_rect", rectArray(BlazePose.rectFromLandmarks(l, width: w, height: h)), doubles(exp["next_rect"]))
}

if failures > 0 {
  print("\(failures) check(s) failed")
  exit(1)
}
print("all BlazePose fixture checks passed")
