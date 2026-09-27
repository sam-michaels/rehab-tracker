# Model card

What the models are, where they came from, and the terms they carry (ADR 0008,
Condition 7). The app ships **BlazePose** (MediaPipe's pose models, Apache-2.0; see
[Public release](#public-release-mediapipe-and-data-for-a-self-trained-model)). The RTMDet and
RTMPose models below are research-only and stay in the harness. Measured performance lives in
[the inference spike report](reports/2026-09-22-inference-spike.md). Conversion provenance
(checkpoint URLs and hashes, toolchain, artifact hashes) is in `manifest.json`, written by
[`convert/convert.py`](convert/convert.py) and published with each release.

| | Detector | Pose |
|---|---|---|
| Artifact | `rtmdet-nano-person-320-fp16` | `rtmpose-s-wholebody-256x192-fp16` |
| Checkpoint | `rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth` | `rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth` |
| Publisher | OpenMMLab ([RTMPose model zoo](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)) | OpenMMLab (same) |
| Training corpus | COCO 2017 (person) + Objects365 (person subset) | COCO-WholeBody + UBody, DWPose distillation |
| **Licence (manifest)** | `LicenseRef-research-only` | `CC-BY-NC-SA-4.0` |

## Licence: non-commercial research use only

**The weights are not Apache-2.0.** This repository's code is Apache-2.0 ([LICENSE](../LICENSE)),
and so is MMPose's. Neither licence covers the model weights. Their terms come from the data
they were trained on. Both models are used and redistributed here **for non-commercial
research only**. The converted pose model is distributed under **CC BY-NC-SA 4.0**, with the
attribution below.

This closes ADR 0002, Condition 3. It was verified on 2026-09-25 against these primary
sources:

| Corpus | Terms, quoted | Source |
|---|---|---|
| **UBody** (pose) | Annotations "belong to the International Digital Economy Academy (IDEA) and are licensed under the Attribution-Non Commercial-Share Alike 4.0 International License (CC-BY-NC-SA 4.0)". The access form: "This data is made available only for noncommercial research purposes." | [Data license](https://docs.google.com/document/d/1R-nn6qguO0YDkPKBleZ8NyrqGrjLXfJ7AQTTsATMYZc/edit), [access form](https://docs.google.com/forms/d/e/1FAIpQLSehgBP7wdn_XznGAM2AiJPiPLTqXXHw5uX9l7qeQ1Dh9HoO_A/viewform) (linked from [IDEA-Research/OSX](https://github.com/IDEA-Research/OSX)) |
| **COCO-WholeBody** (pose) | "COCO-WholeBody dataset is **ONLY** for research and non-commercial use." Commercial use requires contacting the authors. | [jin-s13/COCO-WholeBody, Terms of Use](https://github.com/jin-s13/COCO-WholeBody#terms-of-use) |
| **Objects365** (detector) | "The Objects365 dataset is available for the academic purpose only." Annotations are CC BY 4.0, and "You will NOT distribute the above images." | [objects365.org/download](https://www.objects365.org/download.html) |
| **COCO 2017** (both) | Annotations are CC BY 4.0. Images: "Use of the images must abide by the Flickr Terms of Use." | [cocodataset.org Terms of Use](https://cocodataset.org/#termsofuse) |
| OpenMMLab weights | No terms are stated for published checkpoints. The repository licence (Apache-2.0) covers the code. | [mmpose README, License](https://github.com/open-mmlab/mmpose#license) |
| DWPose | Code is Apache-2.0. No separate terms are stated for its weights or its distillation output. | [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose) |

### What is not established

- **Whether weights are "adapted material" of their training data** under CC licences, or are
  covered by copyright at all. No court or licensor has settled this. This card takes the
  conservative reading: the most restrictive corpus sets the terms, and for the pose model
  that is UBody's ShareAlike clause.
- **Whether distillation changes anything.** The pose model was distilled (DWPose) from a
  teacher trained on the same COCO-WholeBody + UBody data. Distillation does not remove that
  data's terms, and nothing in the sources suggests it does.
- **COCO-WholeBody's own text is inconsistent.** It says "Creative Commons Attribution 4.0"
  but links to the `by-nc/4.0` legal code, and in any case says research and non-commercial
  use only. This card follows the restrictive reading.
- **The detector has no SPDX identifier.** Objects365's "academic purpose only" is not a
  standard licence, so the manifest records it as `LicenseRef-research-only`.

### Why not swap to a permissive checkpoint

ADR 0002 suggested preferring a COCO-WholeBody-only checkpoint. That does not help: every
RTMPose-WholeBody checkpoint is trained on COCO-WholeBody, which is itself non-commercial.
The plain (non-UBody) checkpoints are also AI Challenger-pretrained (`pt-aic-coco`, from
RTMPose-m upwards). No 133-keypoint model in the zoo has permissive training data. A
permissive pose model means giving up whole-body keypoints. For the public App Store build
that is the choice taken; see below and the ADR 0002 amendment of 2026-09-25.

### Obligations when using or redistributing these weights

- Non-commercial research use only.
- Credit the corpora: "UBody courtesy of the International Digital Economy Academy (IDEA)",
  COCO-WholeBody (SenseTime Research), COCO Consortium, Objects365 Consortium, and
  OpenMMLab RTMPose/RTMDet and DWPose for the checkpoints.
- Redistribute the converted pose model under CC BY-NC-SA 4.0, noting that it was converted
  to Core ML at FP16.
- Any commercial use needs permission from the dataset owners. It would also need this card
  revisited: the ADR 0002 alternatives, or retraining on data you are licensed to use.

## Public release: MediaPipe, and data for a self-trained model

The RTMPose and RTMDet weights above stay in the research harness. The App Store build
ships **MediaPipe's pose landmarker models** instead (ADR 0002, amendment of 2026-09-25), run
directly rather than through the MediaPipe SDK.

| | |
|---|---|
| Source | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task` (pinned version `1`, not `latest`) |
| sha256 | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` (the `latest` bundle's zip differs, but its two `.tflite` files are byte-identical) |
| Contents | `pose_detector.tflite` (224×224 → 2254 anchors × 12 + scores), `pose_landmarks_detector.tflite` (256×256 → 39 × 5 landmarks, pose flag, segmentation, 64×64×39 heatmap, world) |
| Fetched by | [`convert/fetch_blazepose.sh`](convert/fetch_blazepose.sh), which verifies the hash; not in git or in our releases |
| Runtime | `TensorFlowLiteSwift` 2.17.0 (Metal delegate, CPU fallback) |

**Why not the MediaPipe SDK.** Google's
[MediaPipe Tasks privacy notice](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice)
says the Tasks APIs "send metrics about the performance and utilization of the APIs in your
app to Google", and that you are "responsible for obtaining informed consent from your app
users". It documents no off switch. The app runs the models with the TensorFlow Lite runtime
instead. Checked on 2026-09-27: the `TensorFlowLiteC` 2.17.0 binaries (core, Metal, Core ML)
contain no URLs other than two tensorflow.org documentation links in error strings, and import
no networking APIs. The only `NSURL` use is the Core ML delegate's file paths.
`delegates/telemetry.cc` reports to the in-process profiler, not over the network.

**Pre/post-processing** is reimplemented from MediaPipe's calculators in
[`runners/blazepose.py`](runners/blazepose.py) (the reference, which cites each source).
The app's Swift port is checked against it via `shared/fixtures/blazepose/`. Compared with the
MediaPipe Python SDK (0.10.18, CPU, IMAGE mode) on the three COCO reference images, the
reference's landmarks differ by a median of **1.9, 4.2 and 5.6 px** (max 8.7, 12.6 and
18.3 px; images 640×425, 500×333 and 640×392). The model is very sensitive to crop details:
shifting the ROI by 1% of the image width moved landmarks by up to 58 px. Changing the crop's
pixel-centre convention helped one image and hurt another. The remaining gap is attributed to
resampling inside the SDK, not established further. The on-device crop (Core Image) matches the
reference's crop within 3/255 per channel, but only with `highQualityDownsample: false`; Core
Image's default prefilter moved it up to 120/255 off.

| | Terms, quoted | Source |
|---|---|---|
| MediaPipe BlazePose GHUM 3D (lite/full/heavy) | "LICENSED UNDER Apache License, Version 2.0". Training data: "images, including consented images (30K), of people using a mobile AR application captured with smartphone cameras"; "The majority of training images (85K) capture a wide range of fitness poses." | [Model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), [Pose Landmarker docs](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) |

Before shipping, note two things. The same model card says: "The model is not intended for
human life-critical decisions. The primary intended application is entertainment." And the
Pose Landmarker bundle also contains a person detector, which has no separate model card
among the sources checked. It is distributed with the landmarker, but its terms are not
independently confirmed here. Apple Vision (`VNDetectHumanRectanglesRequest`) avoids the
question by shipping no weights.

### Permissively licensed training data

This matters only if MediaPipe's accuracy proves insufficient and the project trains its own
model. It was checked on 2026-09-25.

| Dataset | Annotations | Images | Usable? |
|---|---|---|---|
| COCO 2017 keypoints (17 body points) | CC BY 4.0 | Per-image Flickr licence | **Filtered subset only**, see counts below |
| [CMU foot keypoints](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/) (heel, big toe, small toe, on COCO) | CC BY 4.0 | COCO's, same filter | Yes on the filtered subset, but the download host (`posefs1.perception.cs.cmu.edu`) no longer resolves and no mirror was found; a copy must be requested from CMU |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) (person boxes, for a detector) | CC BY 4.0 | "listed as having a CC BY 2.0 license", no warranty | Yes |
| COCO-WholeBody, UBody, Objects365 | Non-commercial or academic only (above) | | No |
| [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html) | CC BY 4.0 | AIST Dance DB: "may not be used for any purpose other than academic research" ([terms](https://aistdancedb.ongaaccel.jp/terms_of_use/)) | No |
| Halpe-FullBody, MPII | No licence found on their project pages | HICO-DET / YouTube | No |
| Own consented recordings | Project-owned | Project-owned | Yes: also the goniometer validation set |

**COCO keypoint images by the image's own licence** (train2017, counted from
`person_keypoints_train2017.json`; val2017 follows the same proportions):

| Image licence | Images with keypoints | People with an ankle labelled | For a public model |
|---|---:|---:|---|
| CC BY | 9,035 | 12,601 | Yes |
| No known copyright restrictions | 296 | 685 | Yes |
| CC BY-SA | 5,040 | 6,862 | Only if the weights are released CC BY-SA |
| CC BY-ND | 2,838 | 3,887 | No (training is arguably adaptation) |
| CC BY-NC / BY-NC-SA / BY-NC-ND | 39,390 | 54,786 | No |

The safe subset is about **9.3k images and 13.3k people with an ankle labelled**, or 14.4k
images and 20k people if you accept CC BY-SA on the weights. Foot labels come only from CMU.
If the CMU foot set follows COCO's licence mix (an estimate: the files could not be
downloaded), about 2.3k people with foot labels remain usable. That is thin for heel and toe,
so own recordings are part of the plan, not an extra.

**Not established:**
- A pretrained backbone brings its own data licence. ImageNet, for example, is
  non-commercial, so any backbone needs the same check.
- COCO's licence tags record Flickr's labels at collection time and are not guaranteed.
- CC licences cover copyright only. They give no consent from the people pictured.
- Canada has no text-and-data-mining exception, and whether fair dealing covers model
  training is unresolved. This card therefore relies on explicit licences only.
