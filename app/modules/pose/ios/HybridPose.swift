import CoreImage
import NitroModules
import QuartzCore
import TensorFlowLite
import VisionCamera

/// The two .tflite models from MediaPipe's pose_landmarker_full.task, run directly: no
/// MediaPipe SDK, because the SDK reports usage metrics to Google (ADR 0002, amendment
/// 2026-09-25). Placed by ml/convert/fetch_blazepose.sh. Output tensors are addressed by
/// name order (Identity, Identity_1, ...), which is how MediaPipe splits them too.
private final class Models {
  let detector: Interpreter
  let landmarks: Interpreter
  let detOut: [Int]  // indices of Identity (boxes), Identity_1 (scores)
  let lmOut: [Int]  // Identity (landmarks), Identity_1 (flag), _2 (segmentation), _3 (heatmap), _4 (world)

  init(bundle: Bundle) throws {
    detector = try Models.load("pose_detector", bundle: bundle)
    landmarks = try Models.load("pose_landmarks_detector", bundle: bundle)
    detOut = try Models.outputOrder(detector)
    lmOut = try Models.outputOrder(landmarks)
  }

  /// GPU (Metal, full precision) where the delegate accepts the graph, CPU otherwise.
  private static func load(_ name: String, bundle: Bundle) throws -> Interpreter {
    guard let path = bundle.path(forResource: name, ofType: "tflite") else {
      throw RuntimeError.error(
        withMessage:
          "pose: model '\(name).tflite' not found in app bundle. Run ml/convert/fetch_blazepose.sh "
          + "(unpacks it into app/modules/pose/ios/models/) and rebuild.")
    }
    var options = Interpreter.Options()
    options.threadCount = 2
    let interpreter =
      try (try? Interpreter(modelPath: path, options: options, delegates: [MetalDelegate()]))
      ?? Interpreter(modelPath: path, options: options)
    try interpreter.allocateTensors()
    return interpreter
  }

  private static func outputOrder(_ it: Interpreter) throws -> [Int] {
    let names = try (0..<it.outputTensorCount).map { try it.output(at: $0).name }
    return names.indices.sorted { names[$0] < names[$1] }
  }
}

/// `pose` frame processor plugin: BlazePose detector -> rotated ROI crop -> landmark model,
/// with MediaPipe's tracking (the next ROI comes from this frame's auxiliary landmarks; the
/// detector re-runs only when the pose is lost; ADR 0002 C2). Geometry is in
/// BlazePoseGeometry.swift. See app/modules/pose/src/Pose.nitro.ts for the JS-facing contract.
public class HybridPose: HybridPoseSpec {
  private static let lock = NSLock()
  private static var models: Models?

  /// Tracking state. Frames arrive one at a time on the camera thread.
  private var rect: PoseRect?
  // No colour management: the models were trained on plain sRGB bytes.
  private let ciContext = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
  private var cropBuffers: [Int: CVPixelBuffer] = [:]
  private var tensorBuffers: [Int: [Float]] = [:]

  /// Loads both models off the JS thread at creation (app launch), so the first camera frame
  /// doesn't stall on interpreter/delegate setup. The getter's lock makes this race-safe.
  override public init() {
    super.init()
    DispatchQueue.global(qos: .userInitiated).async {
      _ = try? HybridPose.getModels()
    }
  }

  public func run(frame: any HybridFrameSpec, forceDetect: Bool?) throws -> PoseResult {
    guard let nativeFrame = frame as? NativeFrame,
      let sampleBuffer = nativeFrame.sampleBuffer,
      let pixelBuffer = sampleBuffer.imageBuffer
    else {
      throw RuntimeError.error(withMessage: "pose: Frame has no CVPixelBuffer")
    }
    let m = try HybridPose.getModels()
    // The upright image, origin at 0,0: every coordinate in the contract refers to it.
    let oriented = CIImage(cvPixelBuffer: pixelBuffer).oriented(HybridPose.cgOrientation(for: frame.orientation))
    let image = oriented.transformed(
      by: CGAffineTransform(translationX: -oriented.extent.minX, y: -oriented.extent.minY))
    let width = Double(image.extent.width), height = Double(image.extent.height)

    var detBox: [Double]?
    var detScore: Double?
    var detMs: Double?
    if forceDetect == true || rect == nil {
      let start = CACurrentMediaTime()
      let input = tensor(
        image, BlazePose.detectorRect(width: width, height: height), size: BlazePose.detSize, range: (-1, 1))
      try m.detector.copy(input, toInputAt: 0)
      try m.detector.invoke()
      let found = BlazePose.bestDetection(
        rawBoxes: try floats(m.detector, m.detOut[0]), rawScores: try floats(m.detector, m.detOut[1]))
      detMs = (CACurrentMediaTime() - start) * 1000
      guard let d = found else {
        rect = nil
        return HybridPose.empty(detScore: 0, detMs: detMs)
      }
      detScore = d.score
      let p0 = BlazePose.unletterbox((d.box[0], d.box[1]), width: width, height: height)
      let p1 = BlazePose.unletterbox((d.box[2], d.box[3]), width: width, height: height)
      detBox = [p0.0, p0.1, p1.0 - p0.0, p1.1 - p0.1]
      rect = BlazePose.rectFromDetection(d, width: width, height: height)
    }
    guard let roi = rect else { return HybridPose.empty(detScore: detScore, detMs: detMs) }

    let poseStart = CACurrentMediaTime()
    try m.landmarks.copy(tensor(image, roi, size: BlazePose.lmSize, range: (0, 1)), toInputAt: 0)
    try m.landmarks.invoke()
    let lms = BlazePose.decodeLandmarks(
      raw: try floats(m.landmarks, m.lmOut[0]), flag: try floats(m.landmarks, m.lmOut[1])[0],
      heatmap: try floats(m.landmarks, m.lmOut[3]), rect: roi)
    let poseMs = (CACurrentMediaTime() - poseStart) * 1000

    guard lms.presence >= BlazePose.posePresence else {
      rect = nil  // lost: re-detect on the next frame
      return HybridPose.empty(detScore: detScore, detMs: detMs, poseMs: poseMs)
    }
    rect = BlazePose.rectFromLandmarks(lms, width: width, height: height)
    let (keypoints, scores) = BlazePose.toWholeBody(lms)
    return PoseResult(
      keypoints: keypoints, scores: scores, detBox: detBox, detScore: detScore, detMs: detMs, poseMs: poseMs)
  }

  // MARK: - Models

  /// Models live in the "Pose.bundle" resource bundle CocoaPods produces from
  /// ios/models/*.tflite. Falls back to the framework bundle itself in case resource_bundles
  /// isn't used by the consuming project's linkage mode.
  private static func resourceBundle() -> Bundle {
    let frameworkBundle = Bundle(for: HybridPose.self)
    if let url = frameworkBundle.url(forResource: "Pose", withExtension: "bundle"),
      let bundle = Bundle(url: url)
    {
      return bundle
    }
    return frameworkBundle
  }

  private static func getModels() throws -> Models {
    lock.lock()
    defer { lock.unlock() }
    if let m = models { return m }
    let m = try Models(bundle: resourceBundle())
    models = m
    return m
  }

  private func floats(_ it: Interpreter, _ index: Int) throws -> [Float] {
    try it.output(at: index).data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }

  // MARK: - Crop

  /// Samples `rect` of the upright image into a size x size RGB float tensor scaled to `range`,
  /// zero outside the image. Same mapping as crop() in ml/runners/blazepose.py: output pixel
  /// centre (u + 0.5) / size goes through the rect's rotation, the inverse of
  /// BlazePose.project. Core Image is bottom-left origin, so y is flipped on both sides.
  private func tensor(_ image: CIImage, _ rect: PoseRect, size: Int, range: (Float, Float)) -> Data {
    let w = Double(image.extent.width), h = Double(image.extent.height), n = Double(size)
    let sx = rect.w * w, sy = rect.h * h
    let c = cos(rect.rotation), s = sin(rect.rotation)
    // Output (Core Image coords) -> source (Core Image coords).
    let toSource = CGAffineTransform(
      a: sx * c / n, b: -sy * s / n, c: sx * s / n, d: sy * c / n,
      tx: w * rect.cx - sx * (c + s) / 2, ty: h - h * rect.cy + sy * (s - c) / 2)
    let bounds = CGRect(x: 0, y: 0, width: size, height: size)
    // Plain bilinear, as the reference and MediaPipe sample. Core Image's default prefilters
    // big downscales (a camera frame -> 224 is ~8x), which moved crops up to 120/255 off.
    // Over black: Core Image only writes where the image has pixels, and the buffer is reused,
    // so padding (the detector's letterbox, ROIs past the frame edge) would keep old frames.
    let crop = image.transformed(by: toSource.inverted(), highQualityDownsample: false)
      .composited(over: CIImage(color: .black)).cropped(to: bounds)

    let buffer = cropBuffer(size)
    ciContext.render(crop, to: buffer, bounds: bounds, colorSpace: nil)

    var out = tensorBuffers[size] ?? [Float](repeating: 0, count: size * size * 3)
    let scale = (range.1 - range.0) / 255, offset = range.0
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<size {
      let row = base + y * rowBytes
      for x in 0..<size {
        let px = row + x * 4, o = (y * size + x) * 3  // BGRA -> RGB
        out[o] = Float(px[2]) * scale + offset
        out[o + 1] = Float(px[1]) * scale + offset
        out[o + 2] = Float(px[0]) * scale + offset
      }
    }
    CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    tensorBuffers[size] = out
    return out.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  private func cropBuffer(_ size: Int) -> CVPixelBuffer {
    if let b = cropBuffers[size] { return b }
    var b: CVPixelBuffer?
    CVPixelBufferCreate(
      nil, size, size, kCVPixelFormatType_32BGRA,
      [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &b)
    cropBuffers[size] = b!
    return b!
  }

  // MARK: - Helpers

  private static func empty(detScore: Double?, detMs: Double?, poseMs: Double = 0) -> PoseResult {
    PoseResult(
      keypoints: [Double](repeating: 0, count: BlazePose.numWholeBody * 2),
      scores: [Double](repeating: 0, count: BlazePose.numWholeBody),
      detBox: nil, detScore: detScore, detMs: detMs, poseMs: poseMs)
  }

  private static func cgOrientation(for o: CameraOrientation) -> CGImagePropertyOrientation {
    switch o {
    case .up: return .up
    case .right: return .right
    case .down: return .down
    case .left: return .left
    }
  }
}
