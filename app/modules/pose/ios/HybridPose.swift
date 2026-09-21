import AVFoundation
import CoreML
import NitroModules
import Vision
import VisionCamera

/// Normalized rect, top-left origin, [0,1] in the Frame buffer's coordinate
/// space. This is the coordinate space of the plugin's public contract
/// (`roi` param, `detBox`, and decoded keypoints) everywhere in this file
/// EXCEPT where explicitly converted to Vision's own bottom-left-origin
/// `regionOfInterest` just before a request is built.
private struct NormRect {
  var x: Double
  var y: Double
  var w: Double
  var h: Double
}

/// One loaded Core ML model plus the pixel size Vision must scale its input to.
private final class LoadedModel {
  let vnModel: VNCoreMLModel
  let inputSize: CGSize
  init(vnModel: VNCoreMLModel, inputSize: CGSize) {
    self.vnModel = vnModel
    self.inputSize = inputSize
  }
}

/// `pose` frame processor plugin: RTMDet-nano person detector + RTMPose-s
/// wholebody, both Core ML, both run through Vision so Vision does the
/// crop/scale/YUV->RGB. See app/modules/pose/src/Pose.nitro.ts for the
/// JS-facing contract.
public class HybridPose: HybridPoseSpec {
  private static let lock = NSLock()
  private static var detectorModel: LoadedModel?
  private static var poseModel: LoadedModel?

  private static let detectorName = "rtmdet-nano-person-320-fp16"
  private static let poseName = "rtmpose-s-wholebody-256x192-fp16"
  private static let detectorInputSize = CGSize(width: 320, height: 320)
  private static let poseInputSize = CGSize(width: 192, height: 256)
  private static let numKeypoints = 133

  public func run(frame: any HybridFrameSpec, roi: [Double]?) throws -> PoseResult {
    guard let nativeFrame = frame as? NativeFrame,
      let sampleBuffer = nativeFrame.sampleBuffer,
      let pixelBuffer = sampleBuffer.imageBuffer
    else {
      throw RuntimeError.error(withMessage: "pose: Frame has no CVPixelBuffer")
    }
    let orientation = HybridPose.cgOrientation(for: frame.orientation)

    var detBox: NormRect?
    var detScore: Double?
    var detMs: Double?
    let poseROI: NormRect

    if let roi = roi {
      guard roi.count == 4 else {
        throw RuntimeError.error(withMessage: "pose: roi must be [x,y,w,h], got \(roi.count) values")
      }
      poseROI = NormRect(x: roi[0], y: roi[1], w: roi[2], h: roi[3])
    } else {
      let start = CACurrentMediaTime()
      let (box, score) = try HybridPose.runDetector(pixelBuffer: pixelBuffer, orientation: orientation)
      detMs = (CACurrentMediaTime() - start) * 1000
      detBox = box
      detScore = score
      poseROI = HybridPose.expandToPoseROI(box)
    }

    let poseStart = CACurrentMediaTime()
    let (keypoints, scores) = try HybridPose.runPose(
      pixelBuffer: pixelBuffer, orientation: orientation, roi: poseROI)
    let poseMs = (CACurrentMediaTime() - poseStart) * 1000

    return PoseResult(
      keypoints: keypoints,
      scores: scores,
      detBox: detBox.map { [$0.x, $0.y, $0.w, $0.h] },
      detScore: detScore,
      detMs: detMs,
      poseMs: poseMs
    )
  }

  // MARK: - Model loading

  /// Compiled models live in the "Pose.bundle" resource bundle CocoaPods
  /// produces from ios/models/*.mlpackage (Xcode compiles .mlpackage ->
  /// .mlmodelc at build time). Falls back to the framework bundle itself in
  /// case resource_bundles isn't used by the consuming project's linkage mode.
  private static func resourceBundle() -> Bundle {
    let frameworkBundle = Bundle(for: HybridPose.self)
    if let url = frameworkBundle.url(forResource: "Pose", withExtension: "bundle"),
      let bundle = Bundle(url: url)
    {
      return bundle
    }
    return frameworkBundle
  }

  private static func loadModel(name: String, inputSize: CGSize) throws -> LoadedModel {
    guard let url = resourceBundle().url(forResource: name, withExtension: "mlmodelc") else {
      throw RuntimeError.error(
        withMessage:
          "pose: compiled model '\(name).mlmodelc' not found in app bundle. "
          + "Place \(name).mlpackage at app/modules/pose/ios/models/ and rebuild "
          + "(Xcode compiles .mlpackage -> .mlmodelc at build time).")
    }
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let mlModel = try MLModel(contentsOf: url, configuration: config)
    let vnModel = try VNCoreMLModel(for: mlModel)
    return LoadedModel(vnModel: vnModel, inputSize: inputSize)
  }

  private static func getDetector() throws -> LoadedModel {
    lock.lock()
    defer { lock.unlock() }
    if let m = detectorModel { return m }
    let m = try loadModel(name: detectorName, inputSize: detectorInputSize)
    detectorModel = m
    return m
  }

  private static func getPoseModel() throws -> LoadedModel {
    lock.lock()
    defer { lock.unlock() }
    if let m = poseModel { return m }
    let m = try loadModel(name: poseName, inputSize: poseInputSize)
    poseModel = m
    return m
  }

  // MARK: - Inference

  private static func runDetector(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation)
    throws -> (NormRect, Double)
  {
    let model = try getDetector()
    let request = VNCoreMLRequest(model: model.vnModel)
    request.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
    try handler.perform([request])

    guard let box = featureValue(request.results, named: "box"), box.count == 4,
      let score = featureValue(request.results, named: "score"), score.count >= 1
    else {
      throw RuntimeError.error(withMessage: "pose: detector output missing 'box'/'score'")
    }
    // box is xyxy in the detector's 320x320 input px (top-left origin raster space).
    let x0 = box[0].doubleValue, y0 = box[1].doubleValue
    let x1 = box[2].doubleValue, y1 = box[3].doubleValue
    let full = NormRect(x: 0, y: 0, w: 1, h: 1)
    let topLeft = mapModelPointToFrame(x: x0, y: y0, inputSize: model.inputSize, roi: full)
    let bottomRight = mapModelPointToFrame(x: x1, y: y1, inputSize: model.inputSize, roi: full)
    let rect = NormRect(
      x: topLeft.0, y: topLeft.1,
      w: bottomRight.0 - topLeft.0, h: bottomRight.1 - topLeft.1)
    return (rect, score[0].doubleValue)
  }

  private static func runPose(
    pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation, roi: NormRect
  ) throws -> ([Double], [Double]) {
    let model = try getPoseModel()
    let request = VNCoreMLRequest(model: model.vnModel)
    request.imageCropAndScaleOption = .scaleFill
    request.regionOfInterest = visionROI(from: roi)
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
    try handler.perform([request])

    guard let kp = featureValue(request.results, named: "keypoints"), kp.count == numKeypoints * 2,
      let sc = featureValue(request.results, named: "scores"), sc.count == numKeypoints
    else {
      throw RuntimeError.error(withMessage: "pose: pose model output missing 'keypoints'/'scores'")
    }

    var keypoints = [Double](repeating: 0, count: numKeypoints * 2)
    for i in 0..<numKeypoints {
      let px = kp[[i, 0] as [NSNumber]].doubleValue
      let py = kp[[i, 1] as [NSNumber]].doubleValue
      let mapped = mapModelPointToFrame(x: px, y: py, inputSize: model.inputSize, roi: roi)
      keypoints[i * 2] = mapped.0
      keypoints[i * 2 + 1] = mapped.1
    }
    var scores = [Double](repeating: 0, count: numKeypoints)
    for i in 0..<numKeypoints {
      scores[i] = sc[i].doubleValue
    }
    return (keypoints, scores)
  }

  // MARK: - Geometry

  /// Vision's `regionOfInterest` is normalized [0,1] with BOTTOM-LEFT origin.
  /// Our `roi` (and every other coordinate in this plugin) is top-left origin.
  private static func visionROI(from r: NormRect) -> CGRect {
    CGRect(x: r.x, y: 1 - r.y - r.h, width: r.w, height: r.h)
  }

  /// Maps a point in a model's raster input px (top-left origin, within
  /// `inputSize`) back to normalized top-left-origin frame coordinates,
  /// given the (top-left-origin, normalized) ROI Vision cropped/scaled from.
  private static func mapModelPointToFrame(
    x: Double, y: Double, inputSize: CGSize, roi: NormRect
  ) -> (Double, Double) {
    let nx = x / Double(inputSize.width)
    let ny = y / Double(inputSize.height)
    return (roi.x + nx * roi.w, roi.y + ny * roi.h)
  }

  /// Expands a detector box to the pose model's 3:4 (w:h) aspect with 1.25x
  /// padding around its center, clamped to stay inside the frame.
  private static func expandToPoseROI(_ box: NormRect) -> NormRect {
    let cx = box.x + box.w / 2
    let cy = box.y + box.h / 2
    var w = box.w * 1.25
    var h = box.h * 1.25
    let targetAspect = 3.0 / 4.0  // w:h
    if w / h < targetAspect {
      w = h * targetAspect
    } else {
      h = w / targetAspect
    }
    w = min(w, 1.0)
    h = min(h, 1.0)
    var x = cx - w / 2
    var y = cy - h / 2
    x = max(0, min(x, 1 - w))
    y = max(0, min(y, 1 - h))
    return NormRect(x: x, y: y, w: w, h: h)
  }

  private static func cgOrientation(for o: CameraOrientation) -> CGImagePropertyOrientation {
    switch o {
    case .up: return .up
    case .right: return .right
    case .down: return .down
    case .left: return .left
    }
  }

  private static func featureValue(_ results: [VNObservation]?, named name: String) -> MLMultiArray? {
    guard let results = results else { return nil }
    for case let obs as VNCoreMLFeatureValueObservation in results where obs.featureName == name {
      return obs.featureValue.multiArrayValue
    }
    return nil
  }
}
