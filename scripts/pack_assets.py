#!/usr/bin/env python3
"""
Copy the finished artifacts into app/src/main/assets so they ship inside the APK.

After this runs the app needs nothing from the network, ever. The APK carries:

    mivolo_face_224.onnx                 the age+gender model
    face_detection_yunet_2023mar.onnx    the face detector
    model_meta.json                      input size, normalisation, age decoding
    face_box_calibration.json            YuNet -> MiVOLO box transform
    test.jpg                             one image for the on-device self-test

Usage:
    python scripts/pack_assets.py
    python scripts/pack_assets.py --model models/mivolo_face_224_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TESTDATA_DIR = REPO_ROOT / "testdata"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"

SELF_TEST_ASSET = "test.jpg"


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def copy(source: Path, name: str | None = None) -> None:
    target = ASSETS_DIR / (name or source.name)
    shutil.copyfile(source, target)
    print(f"  {target.name:38} {human(target.stat().st_size):>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None, help="Override the ONNX model to package.")
    parser.add_argument(
        "--self-test-image",
        type=Path,
        default=None,
        help="Image to bundle as assets/test.jpg. Defaults to the first file in testdata/.",
    )
    args = parser.parse_args()

    meta_path = MODELS_DIR / "model_meta.json"
    if not meta_path.exists():
        sys.exit("models/model_meta.json missing. Run: python scripts/export_model.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_path = args.model or (MODELS_DIR / meta["model_file"])
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    detector_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if not detector_path.exists():
        sys.exit("models/face_detection_yunet_2023mar.onnx missing. Run: python scripts/download_models.py")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Packaging into {ASSETS_DIR}\n")

    # If a different model file was requested, model_meta.json has to agree with
    # it, because Kotlin resolves the asset name through that field.
    if model_path.name != meta["model_file"]:
        meta["model_file"] = model_path.name
        print(f"  (model_meta.json model_file -> {model_path.name})")

    copy(model_path)
    copy(detector_path)

    (ASSETS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  {'model_meta.json':38} {human((ASSETS_DIR / 'model_meta.json').stat().st_size):>10}")

    calibration_path = MODELS_DIR / "face_box_calibration.json"
    if calibration_path.exists():
        copy(calibration_path)
    else:
        print("  face_box_calibration.json              MISSING -- app will use an identity box transform.")
        print("      Run: python scripts/calibrate_face_box.py")

    self_test = args.self_test_image
    if self_test is None and TESTDATA_DIR.exists():
        candidates = sorted(
            p for p in TESTDATA_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self_test = candidates[0] if candidates else None
    if self_test and self_test.exists():
        copy(self_test, SELF_TEST_ASSET)
        print(f"      (self-test image source: {self_test.name})")
    else:
        print("  test.jpg                               MISSING -- the in-app self-test will fail.")
        print("      Run: python scripts/fetch_testdata.py")

    total = sum(p.stat().st_size for p in ASSETS_DIR.iterdir() if p.is_file())
    print(f"\nTotal assets: {human(total)}")
    print("\nNext: build the APK\n    gradlew.bat :app:assembleDebug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
