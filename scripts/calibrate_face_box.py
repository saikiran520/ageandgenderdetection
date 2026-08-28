#!/usr/bin/env python3
"""
Measure how YuNet's face boxes relate to MiVOLO's own detector, and write the
transform that makes them interchangeable.

Why this exists
---------------
MiVOLO feeds the model the *raw* bounding box its YOLOv8x person+face detector
produces -- mivolo/structures.py :: crop_object slices `full_image[y1:y2, x1:x2]`
with no margin and no alignment. The age head is therefore sensitive to how tight
that box is. YuNet is a different detector with a different box convention, so
handing its boxes straight to MiVOLO would put the crop outside the distribution
the model was trained on.

Rather than guessing a margin, this script runs BOTH detectors over the same
photographs, matches their boxes by IoU, and measures the median transform:

    cx' = cx + shift_x * w        w' = w * scale_w
    cy' = cy + shift_y * h        h' = h * scale_h

The result lands in models/face_box_calibration.json, is packaged into the app
assets, and is applied identically by scripts/mivolo_pipeline.py and by
FaceDetector.kt -- so Python and Android crop the same pixels.

The reference detector is NEVER shipped to Android; it is only used here.

Usage:
    python scripts/calibrate_face_box.py
    python scripts/calibrate_face_box.py --images testdata --min-iou 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mivolo_pipeline import (  # noqa: E402
    MODELS_DIR,
    FaceBox,
    YuNetDetector,
    apply_calibration,
    iou,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = REPO_ROOT / "testdata"
REFERENCE_WEIGHTS = MODELS_DIR / "yolov8x_person_face.pt"

# Same thresholds MiVOLO's Detector uses (mivolo/model/yolo_detector.py).
MIVOLO_CONF_THRESH = 0.4
MIVOLO_IOU_THRESH = 0.7


def load_reference_detector():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is required to run calibration.\n"
            "    pip install -r scripts/requirements.txt"
        )
    if not REFERENCE_WEIGHTS.exists():
        sys.exit(
            f"Reference detector not found: {REFERENCE_WEIGHTS}\n"
            "    python scripts/download_models.py --with-detector-reference"
        )
    model = YOLO(str(REFERENCE_WEIGHTS))
    model.fuse()
    return model


def reference_faces(model, image_bgr: np.ndarray) -> List[FaceBox]:
    """Face boxes exactly as MiVOLO's Detector.predict would return them."""
    results = model.predict(
        image_bgr, conf=MIVOLO_CONF_THRESH, iou=MIVOLO_IOU_THRESH, half=False, verbose=False
    )[0]
    names = results.names
    boxes: List[FaceBox] = []
    for det in results.boxes:
        if names[int(det.cls)] != "face":
            continue
        x1, y1, x2, y2 = (float(v) for v in det.xyxy.squeeze().tolist())
        boxes.append(FaceBox(x1, y1, x2, y2, float(det.conf)))
    return boxes


def match_pairs(
    yunet_boxes: List[FaceBox], reference_boxes: List[FaceBox], min_iou: float
) -> List[Tuple[FaceBox, FaceBox]]:
    """Greedy highest-IoU matching, one reference box per YuNet box."""
    pairs: List[Tuple[FaceBox, FaceBox]] = []
    taken: set[int] = set()
    for detected in sorted(yunet_boxes, key=lambda b: b.area, reverse=True):
        best_index, best_iou = -1, min_iou
        for index, reference in enumerate(reference_boxes):
            if index in taken:
                continue
            overlap = iou(detected, reference)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0:
            taken.add(best_index)
            pairs.append((detected, reference_boxes[best_index]))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--min-iou", type=float, default=0.35)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "face_box_calibration.json")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        sys.exit(f"No images in {args.images}. Run: python scripts/fetch_testdata.py")

    print(f"Calibrating over {len(image_paths)} images from {args.images}")
    print(f"  reference : {REFERENCE_WEIGHTS.name}  (official MiVOLO detector)")
    print(f"  candidate : face_detection_yunet_2023mar.onnx\n")

    yunet = YuNetDetector(
        MODELS_DIR / "face_detection_yunet_2023mar.onnx", score_threshold=args.score_threshold
    )
    reference_model = load_reference_detector()

    scale_w: List[float] = []
    scale_h: List[float] = []
    shift_x: List[float] = []
    shift_y: List[float] = []
    matched_images = 0

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        detected = yunet.detect(image)
        reference = reference_faces(reference_model, image)
        pairs = match_pairs(detected, reference, args.min_iou)
        if pairs:
            matched_images += 1
        for candidate, truth in pairs:
            if candidate.width <= 1 or candidate.height <= 1:
                continue
            scale_w.append(truth.width / candidate.width)
            scale_h.append(truth.height / candidate.height)
            shift_x.append((truth.cx - candidate.cx) / candidate.width)
            shift_y.append((truth.cy - candidate.cy) / candidate.height)

    if len(scale_w) < 5:
        sys.exit(
            f"Only {len(scale_w)} matched face pairs -- too few to calibrate. "
            "Add more images to testdata/ and retry."
        )

    calibration = {
        "_comment": (
            "Maps a YuNet face box onto the box MiVOLO's own YOLOv8x person+face detector "
            "would have produced. Generated by scripts/calibrate_face_box.py -- do not edit by hand."
        ),
        "candidate_detector": "face_detection_yunet_2023mar.onnx",
        "reference_detector": "yolov8x_person_face.pt (official MiVOLO detector)",
        "images": len(image_paths),
        "images_with_matches": matched_images,
        "n_samples": len(scale_w),
        "min_iou": args.min_iou,
        "scale_w": round(float(np.median(scale_w)), 5),
        "scale_h": round(float(np.median(scale_h)), 5),
        "shift_x": round(float(np.median(shift_x)), 5),
        "shift_y": round(float(np.median(shift_y)), 5),
        "spread": {
            "scale_w_iqr": [round(float(np.percentile(scale_w, 25)), 4), round(float(np.percentile(scale_w, 75)), 4)],
            "scale_h_iqr": [round(float(np.percentile(scale_h, 25)), 4), round(float(np.percentile(scale_h, 75)), 4)],
            "shift_x_iqr": [round(float(np.percentile(shift_x, 25)), 4), round(float(np.percentile(shift_x, 75)), 4)],
            "shift_y_iqr": [round(float(np.percentile(shift_y, 25)), 4), round(float(np.percentile(shift_y, 75)), 4)],
        },
        "transform": "cx' = cx + shift_x*w ; cy' = cy + shift_y*h ; w' = w*scale_w ; h' = h*scale_h",
    }

    print(f"Matched {len(scale_w)} face pairs across {matched_images}/{len(image_paths)} images\n")
    print(f"  scale_w : {calibration['scale_w']:.4f}   IQR {calibration['spread']['scale_w_iqr']}")
    print(f"  scale_h : {calibration['scale_h']:.4f}   IQR {calibration['spread']['scale_h_iqr']}")
    print(f"  shift_x : {calibration['shift_x']:+.4f}   IQR {calibration['spread']['shift_x_iqr']}")
    print(f"  shift_y : {calibration['shift_y']:+.4f}   IQR {calibration['spread']['shift_y_iqr']}")

    # How much does the calibration actually help? Report mean IoU before/after.
    before, after = [], []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        detected = yunet.detect(image)
        reference = reference_faces(reference_model, image)
        for candidate, truth in match_pairs(detected, reference, args.min_iou):
            before.append(iou(candidate, truth))
            after.append(iou(apply_calibration(candidate, calibration), truth))
    calibration["mean_iou_before"] = round(float(np.mean(before)), 4)
    calibration["mean_iou_after"] = round(float(np.mean(after)), 4)
    print(f"\n  mean IoU vs official detector: {calibration['mean_iou_before']:.4f} "
          f"-> {calibration['mean_iou_after']:.4f}")

    args.output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
