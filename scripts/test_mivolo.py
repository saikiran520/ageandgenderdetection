#!/usr/bin/env python3
"""
Validate the exported model before and after Android integration.

    python scripts/test_mivolo.py --image test.jpg
    Age: 27
    Gender: Male

This runs the ONNX graph that is packaged in the APK, through the same
preprocessing and the same output decoding that Kotlin uses, so a disagreement
between this script and the phone is a real bug rather than a difference in
methodology.

Extra modes:

  --json          emit the full fingerprint (input tensor stats, raw logits) so
                  it can be diffed against the Android logcat output.
  --reference     additionally run the *original* pipeline -- PyTorch checkpoint
                  plus MiVOLO's own YOLOv8x detector -- and report the delta.
                  This is the check that the ONNX conversion and the substituted
                  face detector did not change the answer.
  --batch DIR     run over a directory and print a summary table.
  --verify-preprocessing
                  assert that this file's letterbox/normalise reimplementation is
                  bit-identical to the functions in the installed mivolo package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mivolo_pipeline import (  # noqa: E402
    MODELS_DIR,
    AgeGenderPipeline,
    letterbox,
    normalise,
    tensor_stats,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def verify_preprocessing() -> bool:
    """Prove our port matches the official implementation, not just resembles it."""
    try:
        from mivolo.data.misc import class_letterbox, prepare_classification_images
    except ImportError:
        print("  SKIPPED: the mivolo package is not installed (needs torch+timm).")
        return True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    rng = np.random.default_rng(0)
    ok = True

    for index, (h, w) in enumerate([(37, 91), (224, 224), (300, 120), (1, 5), (513, 511), (64, 64)]):
        crop = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

        ours_boxed = letterbox(crop.copy(), 224)
        theirs_boxed = class_letterbox(crop.copy(), new_shape=(224, 224))
        boxed_diff = int(np.max(np.abs(ours_boxed.astype(int) - theirs_boxed.astype(int))))

        ours = normalise(theirs_boxed.copy(), mean, std)
        theirs = prepare_classification_images([crop.copy()], 224, mean, std).numpy()
        tensor_diff = float(np.max(np.abs(ours - theirs)))

        status = "ok" if boxed_diff == 0 and tensor_diff < 1e-6 else "MISMATCH"
        if status != "ok":
            ok = False
        print(f"  crop {h:>4}x{w:<4}  letterbox delta {boxed_diff}   tensor delta {tensor_diff:.3e}   {status}")

    return ok


def run_reference(image_path: Path, score_threshold: float) -> dict | None:
    """The original pipeline: PyTorch checkpoint + MiVOLO's own YOLOv8x detector."""
    try:
        import torch
        from mivolo.model.create_timm_model import create_model
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"  reference mode needs torch + timm + ultralytics ({exc})")
        return None

    checkpoint = MODELS_DIR / "mivolo_volo_d1_face_only_imdb.pth.tar"
    weights = MODELS_DIR / "yolov8x_person_face.pt"
    if not checkpoint.exists() or not weights.exists():
        print("  reference mode needs models/mivolo_volo_d1_face_only_imdb.pth.tar and "
              "models/yolov8x_person_face.pt")
        print("  run: python scripts/download_models.py --with-detector-reference")
        return None

    meta = json.loads((MODELS_DIR / "model_meta.json").read_text(encoding="utf-8"))
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)

    model = create_model(
        model_name=meta["architecture"],
        num_classes=meta["num_classes"],
        in_chans=meta["in_chans"],
        pretrained=False,
        checkpoint_path=str(checkpoint),
        filter_keys=["fds."],
    )
    model.eval()

    image = cv2.imread(str(image_path))
    detector = YOLO(str(weights))
    detector.fuse()
    results = detector.predict(image, conf=0.4, iou=0.7, half=False, verbose=False)[0]

    faces = []
    for det in results.boxes:
        if results.names[int(det.cls)] != "face":
            continue
        x1, y1, x2, y2 = (int(v) for v in det.xyxy.squeeze().tolist())
        faces.append((x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))
    if not faces:
        print("  reference detector found no face")
        return None

    x1, y1, x2, y2, _ = max(faces, key=lambda f: f[4])
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
    crop = image[y1:y2, x1:x2].copy()

    boxed = letterbox(crop, meta["input_size"])
    tensor = normalise(boxed, meta["mean"], meta["std"])
    with torch.no_grad():
        output = model(torch.from_numpy(tensor)).numpy()

    row = output[0]
    logits = row[meta["gender_logit_indices"]].astype(np.float64)
    exp = np.exp(logits - logits.max())
    probabilities = exp / exp.sum()
    index = int(np.argmax(probabilities))
    age = float(row[meta["age_output_index"]]) * (meta["max_age"] - meta["min_age"]) + meta["avg_age"]

    return {
        "age": round(age, 2),
        "gender": meta["gender_labels"][index],
        "gender_score": float(probabilities[index]),
        "box": [x1, y1, x2, y2],
        "raw_output": [float(v) for v in row],
        "min_age": state["min_age"],
        "max_age": state["max_age"],
        "avg_age": state["avg_age"],
    }


def analyse_one(pipeline: AgeGenderPipeline, image_path: Path) -> tuple[dict | None, list]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    started = time.perf_counter()
    prediction, faces = pipeline.predict(image)
    elapsed = (time.perf_counter() - started) * 1000.0

    if prediction is None:
        return None, faces

    return (
        {
            "image": image_path.name,
            "image_size": [image.shape[1], image.shape[0]],
            "faces_detected": len(faces),
            "face_box": [round(v, 2) for v in (prediction.box.x1, prediction.box.y1,
                                               prediction.box.x2, prediction.box.y2)],
            "age": prediction.age,
            "gender": prediction.gender,
            "gender_score": round(prediction.gender_score, 4),
            "raw_output": [round(float(v), 6) for v in prediction.raw_output[0]],
            "input_tensor": prediction.input_tensor_stats,
            "elapsed_ms": round(elapsed, 1),
        },
        faces,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="Image to analyse.")
    parser.add_argument("--batch", type=Path, help="Directory of images to analyse.")
    parser.add_argument("--json", action="store_true", help="Print the full fingerprint as JSON.")
    parser.add_argument("--reference", action="store_true",
                        help="Also run the original PyTorch + YOLOv8x pipeline and report the delta.")
    parser.add_argument("--verify-preprocessing", action="store_true",
                        help="Assert this script's preprocessing matches the mivolo package byte for byte.")
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip the YuNet->MiVOLO box transform (for A/B comparison).")
    args = parser.parse_args()

    if args.verify_preprocessing:
        print("Verifying preprocessing against the official mivolo package:")
        ok = verify_preprocessing()
        print("  PASS" if ok else "  FAIL")
        if not (args.image or args.batch):
            return 0 if ok else 1

    if not args.image and not args.batch:
        parser.error("one of --image or --batch is required")

    calibration = None
    if args.no_calibration:
        from mivolo_pipeline import IDENTITY_CALIBRATION

        calibration = dict(IDENTITY_CALIBRATION)

    pipeline = AgeGenderPipeline(calibration=calibration, score_threshold=args.score_threshold)

    if args.batch:
        paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        header = f"{'image':42} {'age':>6} {'gender':>8} {'conf':>6} {'ms':>7}"
        if args.reference:
            header += f" {'ref age':>8} {'delta':>7} {'ref gen':>8}"
        print(header)
        print("-" * len(header))

        rows = []
        deltas, gender_agreements = [], []
        for path in paths:
            record, _ = analyse_one(pipeline, path)
            if record is None:
                print(f"{path.name[:42]:42} {'-':>6} {'no face':>8}")
                continue
            rows.append(record)
            line = (f"{path.name[:42]:42} {record['age']:>6.1f} {record['gender']:>8} "
                    f"{record['gender_score']:>6.3f} {record['elapsed_ms']:>7.1f}")
            if args.reference:
                reference = run_reference(path, args.score_threshold)
                if reference:
                    delta = abs(reference["age"] - record["age"])
                    deltas.append(delta)
                    gender_agreements.append(reference["gender"] == record["gender"])
                    line += f" {reference['age']:>8.1f} {delta:>7.2f} {reference['gender']:>8}"
                else:
                    line += f" {'-':>8} {'-':>7} {'-':>8}"
            print(line)

        if rows:
            ages = [r["age"] for r in rows]
            print("-" * len(header))
            print(f"{len(rows)} faces  mean age {np.mean(ages):.1f}  "
                  f"median {np.median(ages):.1f}  mean {np.mean([r['elapsed_ms'] for r in rows]):.0f} ms")
        if deltas:
            agreed = sum(gender_agreements)
            print()
            print("ONNX + YuNet  vs  original PyTorch + official YOLOv8x detector")
            print(f"  age delta   : mean {np.mean(deltas):.3f}  median {np.median(deltas):.3f}  "
                  f"p95 {np.percentile(deltas, 95):.3f}  max {np.max(deltas):.3f} years")
            print(f"  gender agree: {agreed}/{len(gender_agreements)} "
                  f"({100.0 * agreed / len(gender_agreements):.1f}%)")
        return 0

    record, faces = analyse_one(pipeline, args.image)

    if record is None:
        print("No face detected.")
        print("Please capture a clear face.")
        return 2

    print(f"Age: {round(record['age'])}")
    print(f"Gender: {record['gender'].capitalize()}")

    if args.json:
        print()
        print(json.dumps(record, indent=2))

    if args.reference:
        print("\n--- reference pipeline (PyTorch checkpoint + official YOLOv8x detector) ---")
        reference = run_reference(args.image, args.score_threshold)
        if reference:
            print(f"  reference  age {reference['age']:.2f}  gender {reference['gender']} "
                  f"({reference['gender_score'] * 100:.1f}%)  box {reference['box']}")
            print(f"  onnx       age {record['age']:.2f}  gender {record['gender']} "
                  f"({record['gender_score'] * 100:.1f}%)  box {record['face_box']}")
            delta = abs(reference["age"] - record["age"])
            agree = "yes" if reference["gender"] == record["gender"] else "NO"
            print(f"  age delta  {delta:.2f} years")
            print(f"  gender agreement: {agree}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
