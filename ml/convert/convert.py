"""PyTorch -> TorchScript -> Core ML for the two spike models (ADR 0006 C3).

Runs inside the container built from ./Dockerfile (mmdet/mmpose stay there, ADR 0006 C5).
Writes to --out: both .mlpackage (+ zipped, for release upload), manifest.json (checkpoint
provenance, toolchain versions, artifact hashes) and PyTorch reference in/out .npy pairs for
parity.py, which runs separately on the macOS host because Core ML prediction needs macOS.

    python convert.py --out out

Detector wrapper decodes the RTMDet head in-graph (priors + distance decode) and returns the
single highest-score box. ponytail: single-person only; multi-person needs NMS.
Pose wrapper decodes SimCC in-graph (argmax / split_ratio), score = min(x-peak, y-peak), same
as mmpose's own get_simcc_maximum.
"""
import argparse
import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct
import mmcv
import mmdet
import mmengine
import mmpose
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from mmdet.apis import init_detector
from mmdet.structures.bbox import distance2bbox
from mmpose.apis import init_model

MMPOSE_HOME = Path("/opt/mmpose")

DETECTOR = dict(
    name="rtmdet-nano-person-320-fp16",
    config=MMPOSE_HOME / "projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py",
    checkpoint_url="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
                    "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
    input_hw=(320, 320),
    training_corpus="COCO 2017 (person) + Objects365 (person subset)",
)
POSE = dict(
    name="rtmpose-s-wholebody-256x192-fp16",
    config=MMPOSE_HOME / "projects/rtmpose/rtmpose/wholebody_2d_keypoint/"
                         "rtmpose-s_8xb64-270e_coco-wholebody-256x192.py",
    checkpoint_url="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                    "rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth",
    input_hw=(256, 192),
    # RTMPose-s only ships as this DWPose-distilled variant (COCO-WholeBody's plain
    # aic-coco-pretrained checkpoints stop at -m); see final report re: licensing.
    training_corpus="COCO-WholeBody + UBody (DWPose distillation)",
)
# Real photos, not synthetic noise: on a featureless/noisy image the top-1 argmax (no
# clear peak) flips between PyTorch fp32 and Core ML fp16 on rounding alone, which
# breaks the parity gate on a difference that has nothing to do with conversion
# correctness. These COCO samples already ship in the mmpose repo we clone (test
# fixtures), so no extra download or licensing surface for the spike.
REFERENCE_IMAGES = [
    "tests/data/coco/000000000785.jpg",
    "tests/data/coco/000000040083.jpg",
    "tests/data/coco/000000197388.jpg",
]
N_REFERENCE_FRAMES = len(REFERENCE_IMAGES)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return sha256_file(dest)


class DetectorWrapper(nn.Module):
    """image: [1,3,320,320] RGB 0-255 -> box [4] xyxy input-px, score [1]."""

    def __init__(self, det_model):
        super().__init__()
        self.backbone = det_model.backbone
        self.neck = det_model.neck
        self.bbox_head = det_model.bbox_head
        # Trained on BGR (data_preprocessor bgr_to_rgb=False). Rather than permuting
        # channels, apply the BGR-ordered mean/std directly to our RGB input -- same result.
        self.register_buffer("mean", torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        feats = self.neck(self.backbone(x))
        cls_scores, bbox_preds = self.bbox_head(feats)
        priors = self.bbox_head.prior_generator.grid_priors(
            [f.shape[-2:] for f in cls_scores], dtype=x.dtype, device=x.device)
        scores, boxes = [], []
        for cls, box, p in zip(cls_scores, bbox_preds, priors):
            scores.append(cls.permute(0, 2, 3, 1).reshape(-1).sigmoid())
            boxes.append(distance2bbox(p, box.permute(0, 2, 3, 1).reshape(-1, 4)))
        scores, boxes = torch.cat(scores), torch.cat(boxes)
        # index_select (static count=1), not scores[best]: fancy-indexing on a
        # data-dependent scalar traces to a dynamic-shape gather Core ML's runtime
        # then refuses ("data-dependent shapes disabled") and silently mispredicts.
        best = torch.argmax(scores).unsqueeze(0)
        return boxes.index_select(0, best).squeeze(0), scores.index_select(0, best)


class PoseWrapper(nn.Module):
    """image: [1,3,256,192] RGB 0-255 -> keypoints [133,2] input-px, scores [133]."""

    SPLIT_RATIO = 2.0

    def __init__(self, pose_model):
        super().__init__()
        self.backbone = pose_model.backbone
        self.head = pose_model.head
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        pred_x, pred_y = self.head(self.backbone(x))
        x_locs, y_locs = pred_x.argmax(-1).float(), pred_y.argmax(-1).float()
        max_x, max_y = pred_x.amax(-1), pred_y.amax(-1)
        scores = torch.minimum(max_x, max_y)  # mmpose's get_simcc_maximum: min of the two peaks
        keypoints = torch.stack([x_locs, y_locs], dim=-1) / self.SPLIT_RATIO
        return keypoints[0], scores[0]


def precision_for(wrapper_cls):
    """Plain FP16 for the detector; the pose head's linear/matmul (GAU) constants
    overflow fp16 range during conversion-time folding and corrupt SimCC peaks for
    some keypoints (verified: FP32 whole-graph matches PyTorch exactly, plain FP16
    doesn't) -- keep just those ops at FP32, everything else (the CSPNeXt backbone,
    where the compute actually is) stays FP16."""
    if wrapper_cls is DetectorWrapper:
        return ct.precision.FLOAT16
    keep_fp32 = {"linear", "matmul", "einsum"}
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep_fp32)


def convert_one(spec, wrapper_cls, build_model, out_dir, cache_dir):
    ckpt_path = cache_dir / Path(spec["checkpoint_url"]).name
    ckpt_sha256 = download(spec["checkpoint_url"], ckpt_path)

    model = build_model(str(spec["config"]), str(ckpt_path), device="cpu")
    model.eval()
    wrapper = wrapper_cls(model).eval()

    h, w = spec["input_hw"]
    example = torch.randint(0, 256, (1, 3, h, w), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example)

    out_names = ["box", "score"] if wrapper_cls is DetectorWrapper else ["keypoints", "scores"]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, h, w), color_layout="RGB",
                             scale=1.0, bias=[0, 0, 0])],
        outputs=[ct.TensorType(name=n) for n in out_names],
        convert_to="mlprogram",
        compute_precision=precision_for(wrapper_cls),
        minimum_deployment_target=ct.target.iOS17,
    )
    pkg_path = out_dir / f"{spec['name']}.mlpackage"
    mlmodel.save(str(pkg_path))

    zip_path = out_dir / f"{spec['name']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_path.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(pkg_path.parent))
    artifact_sha256 = sha256_file(zip_path)

    ref_dir = out_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    key = "det" if wrapper_cls is DetectorWrapper else "pose"
    for i, rel_path in enumerate(REFERENCE_IMAGES):
        img = np.array(Image.open(MMPOSE_HOME / rel_path).convert("RGB").resize((w, h)),
                       dtype=np.uint8)
        np.save(ref_dir / f"{key}_input_{i}.npy", img)
        with torch.no_grad():
            out0, out1 = wrapper(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float())
        out0_name = "box" if key == "det" else "keypoints"
        out1_name = "score" if key == "det" else "scores"
        np.save(ref_dir / f"{key}_{out0_name}_{i}.npy", out0.numpy())
        np.save(ref_dir / f"{key}_{out1_name}_{i}.npy", out1.numpy())

    return {
        "id": f"{spec['name']}-{artifact_sha256[:8]}",
        "checkpoint_url": spec["checkpoint_url"],
        "checkpoint_sha256": ckpt_sha256,
        "training_corpus": spec["training_corpus"],
        "config": Path(spec["config"]).name,
        "input_resolution_hw": list(spec["input_hw"]),
        "precision": "fp16" if wrapper_cls is DetectorWrapper else "fp16 (linear/matmul kept fp32)",
        "artifact_file": zip_path.name,
        "artifact_sha256": artifact_sha256,
        "reference_frames": N_REFERENCE_FRAMES,
        "output_names": out_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/convert-cache"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    det_manifest = convert_one(DETECTOR, DetectorWrapper, init_detector, args.out, args.cache)
    pose_manifest = convert_one(POSE, PoseWrapper, init_model, args.out, args.cache)

    manifest = {
        "detector": det_manifest,
        "pose": pose_manifest,
        "reference_images": {
            "source": "open-mmlab/mmpose test fixtures (COCO samples), tag v1.3.2",
            "files": REFERENCE_IMAGES,
        },
        "toolchain": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mmengine": mmengine.__version__,
            "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__,
            "mmpose": mmpose.__version__,
            "coremltools": ct.__version__,
        },
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
