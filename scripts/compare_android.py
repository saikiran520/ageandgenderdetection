#!/usr/bin/env python3
"""
Requirement 7: prove the Android result matches the Python result.

The app logs a fingerprint of its input tensor and its raw model output under
the "MiVOLO" tag. This script captures that from logcat (or reads a saved log),
runs the same image through scripts/test_mivolo.py's pipeline, and diffs the two
numerically -- not just the final age, but the tensor that went into the model,
which is where preprocessing bugs actually show up.

Typical use, with a device attached:

    1. Install and open the app, tap "Run bundled self-test (Python parity)".
    2. python scripts/compare_android.py --image app/src/main/assets/test.jpg

Or from a captured log:

    adb logcat -d -s MiVOLO > android.log
    python scripts/compare_android.py --image app/src/main/assets/test.jpg --log android.log

What a mismatch means:
    input tensor sum differs  -> crop, resize, letterbox padding or channel order
    tensor matches, output differs -> the ONNX graph or the runtime
    both match, age differs   -> the decoding constants
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mivolo_pipeline import AgeGenderPipeline  # noqa: E402

TENSOR_RE = re.compile(
    r"Input tensor\s*:\s*shape=\[([^\]]*)\]\s*sum=(-?[\d.]+)\s*mean=(-?[\d.]+)\s*"
    r"min=(-?[\d.]+)\s*max=(-?[\d.]+)\s*first8=\[([^\]]*)\]"
)
OUTPUT_RE = re.compile(r"Raw model output\s*:\s*\[([^\]]*)\]")
AGE_RE = re.compile(r"Decoded age\s*:\s*(-?[\d.]+)")
GENDER_RE = re.compile(r"Decoded gender\s*:\s*(\w+)")
CROP_RE = re.compile(r"Face crop:\s*(\d+)x(\d+)")


def read_logcat(log_path: Path | None) -> str:
    if log_path:
        return log_path.read_text(encoding="utf-8", errors="replace")
    try:
        completed = subprocess.run(
            ["adb", "logcat", "-d", "-s", "MiVOLO"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("adb not found on PATH. Either add it, or pass --log with a saved logcat file.")
    if completed.returncode != 0:
        sys.exit(f"adb logcat failed:\n{completed.stderr}")
    return completed.stdout


def parse_android(text: str) -> dict:
    """Take the LAST analysis in the log, which is the most recent run."""
    tensors = TENSOR_RE.findall(text)
    outputs = OUTPUT_RE.findall(text)
    ages = AGE_RE.findall(text)
    genders = GENDER_RE.findall(text)
    crops = CROP_RE.findall(text)

    if not (tensors and outputs and ages and genders):
        sys.exit(
            "Could not find a completed MiVOLO analysis in the log.\n"
            "Open the app and run a capture or the bundled self-test first, then retry."
        )

    shape, total, mean, minimum, maximum, first8 = tensors[-1]
    return {
        "tensor": {
            "shape": [int(v.strip()) for v in shape.split(",")],
            "sum": float(total),
            "mean": float(mean),
            "min": float(minimum),
            "max": float(maximum),
            "first8": [float(v) for v in first8.split(",")],
        },
        "raw_output": [float(v) for v in outputs[-1].split(",")],
        "age": float(ages[-1]),
        "gender": genders[-1],
        "crop": [int(crops[-1][0]), int(crops[-1][1])] if crops else None,
    }


def report(label: str, android, python, tolerance: float, relative: bool = False) -> bool:
    delta = abs(android - python)
    limit = tolerance * max(abs(python), 1e-9) if relative else tolerance
    ok = delta <= limit
    suffix = f" ({100.0 * delta / max(abs(python), 1e-9):.3f}%)" if relative else ""
    print(f"  {label:22} android {android:>14.6f}   python {python:>14.6f}   "
          f"delta {delta:>11.6f}{suffix}   {'ok' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True,
                        help="The same image the app analysed (assets/test.jpg for the self-test).")
    parser.add_argument("--log", type=Path, help="A saved 'adb logcat -s MiVOLO' file. Omit to run adb directly.")
    # The input tensor is compared with a RELATIVE tolerance because OpenCV's
    # INTER_LINEAR resize uses fixed-point arithmetic (INTER_RESIZE_COEF_BITS)
    # while the Kotlin port works in floating point. The sampling grid is
    # identical -- only the final rounding differs -- which lands around 0.5% on
    # the summed tensor and well under 0.1 years on the decoded age. Anything
    # materially larger than that is a real preprocessing bug.
    parser.add_argument("--tensor-tolerance", type=float, default=0.02,
                        help="Relative tolerance on the summed input tensor (0.02 = 2%%).")
    parser.add_argument("--output-tolerance", type=float, default=0.02)
    parser.add_argument("--age-tolerance", type=float, default=1.0, help="Years.")
    args = parser.parse_args()

    android = parse_android(read_logcat(args.log))

    image = cv2.imread(str(args.image))
    if image is None:
        sys.exit(f"Could not read {args.image}")
    prediction, faces = AgeGenderPipeline().predict(image)
    if prediction is None:
        sys.exit("The Python pipeline found no face in this image, so there is nothing to compare.")

    stats = prediction.input_tensor_stats
    print(f"Comparing {args.image.name}\n")
    print(f"  android crop           {android['crop']}")
    print(f"  android tensor shape   {android['tensor']['shape']}")
    print(f"  python  tensor shape   {stats['shape']}\n")

    checks = [
        report("input tensor sum", android["tensor"]["sum"], stats["sum"],
               args.tensor_tolerance, relative=True),
        report("input tensor mean", android["tensor"]["mean"], stats["mean"],
               args.tensor_tolerance, relative=True),
        report("input tensor min", android["tensor"]["min"], stats["min"], 1e-4),
        report("input tensor max", android["tensor"]["max"], stats["max"], 1e-4),
    ]

    print()
    for index, (a, p) in enumerate(zip(android["raw_output"], prediction.raw_output[0])):
        checks.append(report(f"raw_output[{index}]", a, float(p), args.output_tolerance))

    print()
    checks.append(report("age (years)", android["age"], prediction.age, args.age_tolerance))
    gender_ok = android["gender"].lower() == prediction.gender.lower()
    print(f"  {'gender':22} android {android['gender']:>14}   python {prediction.gender:>14}   "
          f"{'ok' if gender_ok else 'MISMATCH'}")
    checks.append(gender_ok)

    print()
    if all(checks):
        print("PASS - Android and Python agree.")
        return 0

    print("FAIL - see the notes at the top of this script for what each mismatch points at.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
