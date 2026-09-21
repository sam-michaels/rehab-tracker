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
  s.source_files = '*.swift'

  # RTMDet-nano + RTMPose-s-WholeBody .mlpackage, placed manually (not in
  # git, see app/modules/pose/ios/models/, gitignored). Xcode compiles
  # .mlpackage -> .mlmodelc at build time (ADR 0006 C6) because it's a
  # resource_bundle, not a plain source file.
  s.resource_bundles = {
    'Pose' => ['models/*.mlpackage']
  }

  # Absolute path: CocoaPods evaluates `load` relative to its own cwd, not
  # this podspec's directory, for pods referenced via a local :path. nitro.json
  # lives at the module root (one level up from this podspec), so that's
  # where nitrogen wrote nitrogen/generated/.
  load File.join(__dir__, '..', 'nitrogen/generated/ios/Pose+autolinking.rb')
  add_nitrogen_files(s)

  # add_nitrogen_files() assumes the podspec sits next to nitro.json; ours is
  # one level down in ios/, so its generated "nitrogen/..." globs resolve to
  # the wrong place. Rewrite them to point up one directory.
  %i[source_files public_header_files private_header_files].each do |key|
    s.attributes_hash[key.to_s] = Array(s.attributes_hash[key.to_s]).map do |f|
      f.start_with?('nitrogen/') ? "../#{f}" : f
    end
  end
end
