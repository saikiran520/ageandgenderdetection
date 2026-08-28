#!/usr/bin/env python3
"""
Download every open-source artifact this POC needs.

Nothing here is invented: every URL below comes from the official MiVOLO
repository (https://github.com/WildChlamydia/MiVOLO) or from the OpenCV Zoo.
See docs/MODEL_PROVENANCE.md for the full paper trail.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --with-detector-reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -r scripts/requirements.txt")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

GDRIVE_ENDPOINT = "https://drive.usercontent.google.com/download"


@dataclass
class Artifact:
    filename: str
    description: str
    source: str
    license: str
    url: str = ""
    expected_size: Optional[int] = None
    optional: bool = False
    gdrive_id: Optional[str] = None
    notes: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


# ---------------------------------------------------------------------------
# The artifact table.
#
# 1) MiVOLO age+gender checkpoint.
#    Row "volo_d1 / face_only, age, gender / IMDB-cleaned" of the model table in
#    the official MiVOLO README. It is the *face-only* variant, which is exactly
#    what this POC needs: 3-channel input, no person crop, no person detector.
#    README link: https://drive.google.com/file/d/1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP/view
#
# 2) YuNet face detector.
#    MiVOLO ships a YOLOv8x person+face detector (68 M params, ~270 MB as ONNX)
#    which is far too heavy for an APK. We substitute YuNet from the OpenCV Zoo
#    (232 KB) and calibrate its boxes against the official detector -- see
#    scripts/calibrate_face_box.py. Calibration keeps the crop that reaches
#    MiVOLO inside the distribution the model was trained on.
#
# 3) Official YOLOv8x person+face detector (OPTIONAL).
#    Only needed to *derive* the calibration in (2). Never shipped to Android.
# ---------------------------------------------------------------------------

ARTIFACTS = [
    Artifact(
        filename="mivolo_volo_d1_face_only_imdb.pth.tar",
        gdrive_id="1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP",
        description=(
            "MiVOLO volo_d1, face-only, age + gender, trained on IMDB-cleaned. "
            "Age MAE 4.22, CS@5 68.68, gender accuracy 99.38%."
        ),
        source=(
            "Official MiVOLO README model table -> "
            "https://drive.google.com/file/d/1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP/view"
        ),
        license="MiVOLO repository license (https://github.com/WildChlamydia/MiVOLO/blob/main/license)",
        expected_size=103498750,
        notes="Read by scripts/export_model.py, converted to ONNX, then packaged into app assets.",
    ),
    Artifact(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        description="YuNet face detector, 232 KB ONNX, runs on ONNX Runtime on desktop and Android alike.",
        source="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        license="MIT (OpenCV Zoo / libfacedetection)",
        expected_size=232589,
        notes=(
            "Copied verbatim into app/src/main/assets/. Decoded by FaceDetector.kt exactly as OpenCV's "
            "cv::FaceDetectorYN does (strides 8/16/32, score = sqrt(cls*obj))."
        ),
    ),
    Artifact(
        filename="yolov8x_person_face.pt",
        url="https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector/resolve/main/yolov8x_person_face.pt",
        description="Official MiVOLO person+face detector. Reference only -- used to calibrate YuNet's boxes.",
        source="https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector",
        license="AGPL-3.0",
        optional=True,
        notes="NOT shipped to Android. Used by scripts/calibrate_face_box.py only.",
    ),
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_to_file(response, dest: Path) -> None:
    total = int(response.headers.get("Content-Length") or 0)
    done = 0
    tmp = dest.with_name(dest.name + ".part")
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {human(done)} / {human(total)} ({100.0 * done / total:5.1f}%)", end="", flush=True)
            else:
                print(f"\r    {human(done)}", end="", flush=True)
    print()
    if dest.exists():
        dest.unlink()
    tmp.replace(dest)


def download_gdrive(file_id: str, dest: Path) -> None:
    """Google Drive large-file download, including the virus-scan confirm handshake."""
    with requests.Session() as session:
        params = {"id": file_id, "export": "download", "confirm": "t"}
        response = session.get(GDRIVE_ENDPOINT, params=params, stream=True, timeout=60)
        response.raise_for_status()

        if "text/html" in response.headers.get("Content-Type", ""):
            # Drive returned the interstitial page; replay its hidden form fields.
            fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', response.text))
            if not fields:
                raise RuntimeError(
                    "Google Drive returned an HTML page with no download form. The daily quota for "
                    "this file may be exhausted; retry later or download it manually into models/ "
                    "(see docs/MODEL_PROVENANCE.md)."
                )
            response = session.get(GDRIVE_ENDPOINT, params=fields, stream=True, timeout=60)
            response.raise_for_status()

        _stream_to_file(response, dest)


def download_http(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        _stream_to_file(response, dest)


def fetch(artifact: Artifact, force: bool = False) -> bool:
    dest = artifact.path
    if dest.exists() and not force:
        size = dest.stat().st_size
        if artifact.expected_size and size != artifact.expected_size:
            print(f"  ! {dest.name} is {human(size)}, expected {human(artifact.expected_size)} -- re-downloading")
        else:
            print(f"  = {dest.name} already present ({human(size)})")
            return True

    print(f"  > {dest.name}")
    print(f"    {artifact.description}")
    print(f"    source : {artifact.source}")
    print(f"    license: {artifact.license}")
    try:
        if artifact.gdrive_id:
            download_gdrive(artifact.gdrive_id, dest)
        else:
            download_http(artifact.url, dest)
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return False

    size = dest.stat().st_size
    if artifact.expected_size and size != artifact.expected_size:
        print(f"    WARNING: size {human(size)} != expected {human(artifact.expected_size)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--with-detector-reference",
        action="store_true",
        help="Also download the official YOLOv8x person+face detector (~270 MB), needed only to re-run calibration.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {MODELS_DIR}\n")

    ok = True
    manifest = []
    for artifact in ARTIFACTS:
        if artifact.optional and not args.with_detector_reference:
            print(f"  - {artifact.filename} skipped (pass --with-detector-reference to fetch it)\n")
            continue
        if not fetch(artifact, force=args.force):
            ok = False
            continue
        digest = sha256_of(artifact.path)
        print(f"    sha256 : {digest}\n")
        manifest.append(
            {
                "filename": artifact.filename,
                "size": artifact.path.stat().st_size,
                "sha256": digest,
                "description": artifact.description,
                "source": artifact.source,
                "license": artifact.license,
                "notes": artifact.notes,
            }
        )

    if manifest:
        manifest_path = MODELS_DIR / "download_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path}")

    if not ok:
        print("\nOne or more downloads failed. See docs/MODEL_PROVENANCE.md for manual instructions.")
        return 1

    print("\nAll required artifacts present. Next: python scripts/export_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
