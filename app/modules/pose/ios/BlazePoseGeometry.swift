import Foundation

// Pre/post-processing around MediaPipe's BlazePose models, ported line for line from
// ml/runners/blazepose.py (the reference, which cites the MediaPipe calculators each constant
// comes from). Foundation only, so check/main.swift can assert it against
// shared/fixtures/blazepose/ with plain swiftc, on macOS or Linux.
//
// Coordinates: normalized [0,1] in the upright image, origin top-left. A PoseRect is
// MediaPipe's NormalizedRect: w and h normalized to image width and height, rotation in radians.

struct PoseRect: Equatable {
  var cx: Double
  var cy: Double
  var w: Double
  var h: Double
  var rotation: Double
}

struct PoseDetection {
  var score: Double
  var box: [Double]  // xmin, ymin, xmax, ymax in the letterboxed 224 input
  var keypoints: [(Double, Double)]  // 4, same space
}

struct PoseLandmarks {
  var presence: Double
  var points: [(Double, Double)]  // 39, image-normalized
  var visibility: [Double]  // 39
}

enum BlazePose {
  static let detSize = 224
  static let lmSize = 256
  static let numLandmarks = 39
  static let heatmapSize = 64
  static let minDetScore = 0.5
  static let nmsIoU = 0.3
  static let scoreClip = 100.0
  static let roiScale = 1.25
  static let posePresence = 0.5
  static let heatmapKernel = 7
  static let heatmapMinConf = 0.5
  static let numWholeBody = 133

  /// MediaPipe landmark -> COCO-WholeBody slot. Unmapped slots (small toes 18/21, face, hands)
  /// keep score 0. Foot index -> big toe, heel -> heel.
  static let mpToWholeBody: [(Int, Int)] = [
    (0, 0), (2, 1), (5, 2), (7, 3), (8, 4),
    (11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
    (23, 11), (24, 12), (25, 13), (26, 14), (27, 15), (28, 16),
    (31, 17), (29, 19), (32, 20), (30, 22),
  ]

  static func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-min(100, max(-100, x)))) }

  /// SSD anchor centres. Layers sharing a stride are merged; each adds 2 anchors per cell.
  static let anchors: [(Double, Double)] = {
    let strides = [8, 16, 32, 32, 32]
    var out: [(Double, Double)] = []
    var layer = 0
    while layer < strides.count {
      var n = 0
      var last = layer
      while last < strides.count && strides[last] == strides[layer] {
        n += 2
        last += 1
      }
      let fm = Int((Double(detSize) / Double(strides[layer])).rounded(.up))
      for y in 0..<fm {
        for x in 0..<fm {
          let c = ((Double(x) + 0.5) / Double(fm), (Double(y) + 0.5) / Double(fm))
          out.append(contentsOf: repeatElement(c, count: n))
        }
      }
      layer = last
    }
    return out
  }()

  /// rawBoxes [2254 * 12], rawScores [2254] -> MediaPipe's first weighted-NMS output, or nil.
  static func bestDetection(rawBoxes: [Float], rawScores: [Float]) -> PoseDetection? {
    var cands: [PoseDetection] = []
    for (i, raw) in rawScores.enumerated() {
      let score = sigmoid(Double(raw))
      guard score >= minDetScore else { continue }
      let r = { (k: Int) in Double(rawBoxes[i * 12 + k]) / Double(detSize) }
      let (ax, ay) = anchors[i]
      let cx = r(0) + ax, cy = r(1) + ay, w = r(2), h = r(3)
      let kps = (0..<4).map { k in (r(4 + 2 * k) + ax, r(5 + 2 * k) + ay) }
      cands.append(PoseDetection(score: score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], keypoints: kps))
    }
    guard let top = cands.max(by: { $0.score < $1.score }) else { return nil }
    let group = cands.filter { iou($0.box, top.box) > nmsIoU }
    let total = group.reduce(0) { $0 + $1.score }
    var box = [Double](repeating: 0, count: 4)
    var kps = [(Double, Double)](repeating: (0, 0), count: 4)
    for d in group {
      for k in 0..<4 { box[k] += d.box[k] * d.score }
      for k in 0..<4 {
        kps[k].0 += d.keypoints[k].0 * d.score
        kps[k].1 += d.keypoints[k].1 * d.score
      }
    }
    return PoseDetection(
      score: top.score, box: box.map { $0 / total }, keypoints: kps.map { ($0.0 / total, $0.1 / total) })
  }

  static func iou(_ a: [Double], _ b: [Double]) -> Double {
    let ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    let iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    let inter = ix * iy
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return union > 0 ? inter / union : 0
  }

  /// Padding on each side, as a fraction of the square detector input.
  static func letterbox(width: Double, height: Double) -> (Double, Double) {
    let side = max(width, height)
    return ((1 - width / side) / 2, (1 - height / side) / 2)
  }

  static func unletterbox(_ p: (Double, Double), width: Double, height: Double) -> (Double, Double) {
    let (px, py) = letterbox(width: width, height: height)
    return ((p.0 - px) / (1 - 2 * px), (p.1 - py) / (1 - 2 * py))
  }

  /// The whole image, letterboxed to a square: the detector's crop.
  static func detectorRect(width: Double, height: Double) -> PoseRect {
    let side = max(width, height)
    return PoseRect(cx: 0.5, cy: 0.5, w: side / width, h: side / height, rotation: 0)
  }

  static func normalizeRadians(_ a: Double) -> Double {
    a - 2 * Double.pi * ((a + Double.pi) / (2 * Double.pi)).rounded(.down)
  }

  /// AlignmentPointsRects (start 0, end 1, target 90 deg) then RectTransformation (x1.25,
  /// square_long). p0 is the hip centre, p1 the scale/rotation point.
  static func alignmentRect(_ p0: (Double, Double), _ p1: (Double, Double), width: Double, height: Double)
    -> PoseRect
  {
    let x0 = p0.0 * width, y0 = p0.1 * height
    let x1 = p1.0 * width, y1 = p1.1 * height
    let size = 2 * hypot(x1 - x0, y1 - y0) * roiScale
    let rotation = normalizeRadians(Double.pi / 2 - atan2(-(y1 - y0), x1 - x0))
    return PoseRect(cx: p0.0, cy: p0.1, w: size / width, h: size / height, rotation: rotation)
  }

  static func rectFromDetection(_ d: PoseDetection, width: Double, height: Double) -> PoseRect {
    alignmentRect(
      unletterbox(d.keypoints[0], width: width, height: height),
      unletterbox(d.keypoints[1], width: width, height: height), width: width, height: height)
  }

  /// Next frame's ROI from auxiliary landmarks 33 (centre) and 34 (scale/rotation).
  static func rectFromLandmarks(_ l: PoseLandmarks, width: Double, height: Double) -> PoseRect {
    alignmentRect(l.points[33], l.points[34], width: width, height: height)
  }

  /// Crop-normalized point -> image-normalized, through the rect's rotation.
  static func project(_ p: (Double, Double), _ r: PoseRect) -> (Double, Double) {
    let x = p.0 - 0.5, y = p.1 - 0.5
    let c = cos(r.rotation), s = sin(r.rotation)
    return ((c * x - s * y) * r.w + r.cx, (s * x + c * y) * r.h + r.cy)
  }

  /// points: crop-normalized, 39. heatmap: [64 * 64 * 39] raw logits, HWC.
  static func refineFromHeatmap(_ points: [(Double, Double)], heatmap: [Float]) -> [(Double, Double)] {
    let hs = heatmapSize
    let off = (heatmapKernel - 1) / 2
    var out = points
    for (i, (x, y)) in points.enumerated() {
      let col = Int(x * Double(hs)), row = Int(y * Double(hs))  // truncates toward zero, as C++
      guard col >= 0, col < hs, row >= 0, row < hs else { continue }
      var total = 0.0, wc = 0.0, wr = 0.0, peak = 0.0
      for r in max(0, row - off)..<min(hs, row + off + 1) {
        for c in max(0, col - off)..<min(hs, col + off + 1) {
          let conf = sigmoid(Double(heatmap[(r * hs + c) * numLandmarks + i]))
          total += conf
          peak = max(peak, conf)
          wc += Double(c) * conf
          wr += Double(r) * conf
        }
      }
      if peak >= heatmapMinConf && total > 0 {
        out[i] = (wc / Double(hs) / total, wr / Double(hs) / total)
      }
    }
    return out
  }

  /// raw [195], flag (already a probability), heatmap [64 * 64 * 39].
  static func decodeLandmarks(raw: [Float], flag: Float, heatmap: [Float], rect: PoseRect) -> PoseLandmarks {
    let crop = (0..<numLandmarks).map { i in
      (Double(raw[i * 5]) / Double(lmSize), Double(raw[i * 5 + 1]) / Double(lmSize))
    }
    return PoseLandmarks(
      presence: Double(flag),
      points: refineFromHeatmap(crop, heatmap: heatmap).map { project($0, rect) },
      visibility: (0..<numLandmarks).map { sigmoid(Double(raw[$0 * 5 + 3])) })
  }

  /// -> (keypoints [133 * 2] flat x,y, scores [133]) in COCO-WholeBody slots.
  static func toWholeBody(_ l: PoseLandmarks) -> ([Double], [Double]) {
    var kps = [Double](repeating: 0, count: numWholeBody * 2)
    var scores = [Double](repeating: 0, count: numWholeBody)
    for (mp, wb) in mpToWholeBody {
      kps[2 * wb] = l.points[mp].0
      kps[2 * wb + 1] = l.points[mp].1
      scores[wb] = l.visibility[mp]
    }
    return (kps, scores)
  }
}
