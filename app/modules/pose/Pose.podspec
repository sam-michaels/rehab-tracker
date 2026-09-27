Pod::Spec.new do |s|
  s.name           = 'Pose'
  s.version        = '1.0.0'
  s.summary        = 'vision-camera frame processor plugin: MediaPipe BlazePose models via TensorFlow Lite, no MediaPipe SDK'
  s.author         = ''
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.platforms      = { :ios => '17.0' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'VisionCamera'
  s.dependency 'TensorFlowLiteSwift/Metal', '~> 2.17.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  # Only our own implementation here; nitrogen's autolinking.rb (loaded below)
  # adds the generated shared/ios sources itself and knows to skip Android.
  s.source_files = 'ios/*.swift'

  # pose_detector.tflite + pose_landmarks_detector.tflite, placed by
  # ml/convert/fetch_blazepose.sh (not in git; ios/models/ is gitignored). Only .tflite:
  # the research RTMPose .mlpackages fetch.sh puts in the same directory are
  # non-commercial and must not ship (ml/MODEL_CARD.md).
  s.resource_bundles = {
    'Pose' => ['ios/models/*.tflite']
  }

  load File.join(__dir__, 'nitrogen/generated/ios/Pose+autolinking.rb')
  add_nitrogen_files(s)
end
