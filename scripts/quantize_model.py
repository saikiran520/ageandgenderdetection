#!/usr/bin/env python3
"""
Optional: shrink the packaged model with dynamic INT8 quantization.

The POC ships full-precision fp32 (~99 MB) so the Android result matches the
Python result to within float noise. This script produces an INT8 variant
(~25 MB) and, importantly, MEASURES the accuracy cost on your own test images
before you decide to ship it.

    python scripts/quantize_model.py
    python scripts/quantize_model.py --images testdata

Then, if the drift is acceptable:

    python scripts/pack_assets.py --model models/mivolo_face_224_int8.onnx
    gradlew.bat :app:assembleDebug

Note: MiVOLO is a transformer. Dynamic quantization of attention MatMuls is the
usual source of drift, so do look at the reported numbers rather than assuming.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def human(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "testdata",
                        help="Images used to measure the accuracy cost.")
    args = parser.parse_args()

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        sys.exit("onnxruntime is required.\n    pip install -r scripts/requirements.txt")

    meta_path = MODELS_DIR / "model_meta.json"
    if not meta_path.exists():
        sys.exit("models/model_meta.json missing. Run: python scripts/export_model.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    source = args.input or (MODELS_DIR / meta["model_file"])
    target = args.output or source.with_name(source.stem + "_int8.onnx")
    if not source.exists():
        sys.exit(f"Model not found: {source}")

    print(f"Quantizing {source.name} -> {target.name}")
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
    )
    print(f"  fp32 {human(source.stat().st_size)}  ->  int8 {human(target.stat().st_size)}")

    # Measure what it cost, rather than asserting it is fine.
    import cv2

    from mivolo_pipeline import AgeGenderPipeline

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ) if args.images.exists() else []

    if not image_paths:
        print("\nNo test images found, so accuracy drift was NOT measured.")
        print("Run: python scripts/fetch_testdata.py")
        return 0

    print(f"\nMeasuring drift over {len(image_paths)} images")
    fp32 = AgeGenderPipeline()
    int8 = AgeGenderPipeline(
        model_path=target,
        detector_path=MODELS_DIR / "face_detection_yunet_2023mar.onnx",
        meta_path=meta_path,
    )

    age_deltas, gender_matches = [], []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        a, _ = fp32.predict(image)
        b, _ = int8.predict(image)
        if a is None or b is None:
            continue
        age_deltas.append(abs(a.age - b.age))
        gender_matches.append(a.gender == b.gender)

    if not age_deltas:
        print("  no comparable faces found")
        return 0

    print(f"  age drift    : mean {np.mean(age_deltas):.3f}  median {np.median(age_deltas):.3f}  "
          f"p95 {np.percentile(age_deltas, 95):.3f}  max {np.max(age_deltas):.3f} years")
    print(f"  gender agree : {sum(gender_matches)}/{len(gender_matches)} "
          f"({100.0 * sum(gender_matches) / len(gender_matches):.1f}%)")
    print(f"\nIf that is acceptable:\n    python scripts/pack_assets.py --model {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
