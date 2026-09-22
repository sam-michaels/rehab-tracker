Pod::Spec.new do |s|
  s.name           = 'Pose'
  s.version        = '1.0.0'
  s.summary        = 'vision-camera frame processor plugin: RTMDet-nano + RTMPose-WholeBody Core ML inference'
  s.author         = ''
  s.homepage       = 'https://docs.expo.dev/modules/'
  s.platforms      = { :ios => '17.0' }
  s.source         = { git: '' }
  s.static_framework = true

  s.dependency 'VisionCamera'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }

  # Only our own implementation here; nitrogen's autolinking.rb (loaded below)
  # adds the generated shared/ios sources itself and knows to skip Android.
  s.source_files = 'ios/*.swift'

  # RTMDet-nano + RTMPose-s-WholeBody .mlpackage, placed manually (not in
  # git, see ios/models/, gitignored). Xcode compiles
  # .mlpackage -> .mlmodelc at build time (ADR 0006 C6) because it's a
  # resource_bundle, not a plain source file.
  s.resource_bundles = {
    'Pose' => ['ios/models/*.mlpackage']
  }

  load File.join(__dir__, 'nitrogen/generated/ios/Pose+autolinking.rb')
  add_nitrogen_files(s)
end
