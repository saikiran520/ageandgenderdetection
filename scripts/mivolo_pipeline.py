#!/usr/bin/env python3
"""
The reference age/gender pipeline, ONNX-only.

This module is the single source of truth for what the Android app must do.
Every step below has a line-for-line counterpart in Kotlin:

    detect faces (YuNet)          -> FaceDetector.kt
    calibrate + clamp the box     -> FaceDetector.kt / MiVOLOProcessor.kt
    crop, letterbox, normalise    -> ImagePreprocessor.kt
    run MiVOLO, decode outputs    -> MiVOLOProcessor.kt

It deliberately depends only on numpy + opencv + onnxruntime, so it can be run
without torch, timm or the mivolo package installed.

Preprocessing provenance (MiVOLO commit at time of writing):
  * mivolo/data/misc.py :: class_letterbox              -> letterbox() below
  * mivolo/data/misc.py :: prepare_classification_images-> normalise() below
  * mivolo/structures.py :: crop_object                 -> crop_face() below
  * mivolo/model/mi_volo.py :: fill_in_results          -> decode() below
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ASSETS_DIR = REPO_ROOT / "app" / "src" / "main" / "assets"

# YuNet constants, mirroring cv::FaceDetectorYNImpl in
# opencv/modules/objdetect/src/face_detect.cpp
YUNET_STRIDES = (8, 16, 32)
YUNET_DIVISOR = 32
YUNET_OUTPUT_NAMES = [
    "cls_8", "cls_16", "cls_32",
    "obj_8", "obj_16", "obj_32",
    "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
]

# face_detection_yunet_2023mar.onnx declares a fixed input of [1, 3, 640, 640],
# so the detector canvas is always 640x640 and the three grids are always
# 80x80, 40x40 and 20x20. The source image is scaled down to fit (never up) and
# padded bottom/right with zeros, which is the same padding convention OpenCV's
# FaceDetectorYN uses in padWithDivisor.
YUNET_CANVAS = 640
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 50

# Identity calibration -- used when models/face_box_calibration.json is absent.
IDENTITY_CALIBRATION = {
    "scale_w": 1.0,
    "scale_h": 1.0,
    "shift_x": 0.0,
    "shift_y": 0.0,
    "n_samples": 0,
    "note": "identity (calibration not run)",
}


@dataclass
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class Prediction:
    age: float
    gender: str
    gender_score: float
    box: FaceBox
    raw_output: np.ndarray
    input_tensor_stats: dict


# ---------------------------------------------------------------------------
# YuNet face detection
# ---------------------------------------------------------------------------


class YuNetDetector:
    """Faithful re-implementation of cv::FaceDetectorYN on top of ONNX Runtime.

    OpenCV's own class cannot be used here because the identical decoding has to
    exist in Kotlin as well; keeping both sides on raw ONNX Runtime means Python
    and Android execute exactly the same graph and exactly the same maths.
    """

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        canvas: int = YUNET_CANVAS,
    ) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.canvas = canvas

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        src_h, src_w = image_bgr.shape[:2]

        # Only ever downscale, so small captures are not blurred by upsampling.
        scale = min(1.0, self.canvas / float(max(src_h, src_w)))
        if scale < 1.0:
            det_w, det_h = int(round(src_w * scale)), int(round(src_h * scale))
            resized = cv2.resize(image_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_w, det_h = src_w, src_h
            resized = image_bgr

        pad_w = pad_h = self.canvas
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h - det_h, 0, pad_w - det_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # cv::dnn::blobFromImage defaults: scalefactor 1.0, no mean, swapRB=false.
        # So: raw 0..255 values, BGR channel order, NCHW.
        blob = padded.astype(np.float32).transpose(2, 0, 1)[None, ...]
        outputs = self.session.run(YUNET_OUTPUT_NAMES, {self.input_name: blob})
        by_name = dict(zip(YUNET_OUTPUT_NAMES, outputs))

        boxes: List[FaceBox] = []
        for stride in YUNET_STRIDES:
            cls = by_name[f"cls_{stride}"].reshape(-1)
            obj = by_name[f"obj_{stride}"].reshape(-1)
            bbox = by_name[f"bbox_{stride}"].reshape(-1, 4)

            cols, rows = pad_w // stride, pad_h // stride
            scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            keep = np.nonzero(scores >= self.score_threshold)[0]
            for idx in keep:
                r, c = divmod(int(idx), cols)
                if r >= rows:
                    continue
                cx = (c + bbox[idx, 0]) * stride
                cy = (r + bbox[idx, 1]) * stride
                w = math.exp(float(bbox[idx, 2])) * stride
                h = math.exp(float(bbox[idx, 3])) * stride
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                # Undo the detection-time downscale so boxes are in source pixels.
                boxes.append(
                    FaceBox(
                        x1=float(x1 / scale),
                        y1=float(y1 / scale),
                        x2=float((x1 + w) / scale),
                        y2=float((y1 + h) / scale),
                        score=float(scores[idx]),
                    )
                )

        return nms(boxes, self.nms_threshold, self.top_k)


def iou(a: FaceBox, b: FaceBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: List[FaceBox], threshold: float, top_k: int) -> List[FaceBox]:
    ordered = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[FaceBox] = []
    for candidate in ordered:
        if len(kept) >= top_k:
            break
        if all(iou(candidate, k) <= threshold for k in kept):
            kept.append(candidate)
    return kept


def select_primary_face(boxes: Sequence[FaceBox]) -> Optional[FaceBox]:
    """Pick the subject of the photo: the largest face.

    A POC capture has one intended subject, and in a hand-held portrait the
    subject is nearer the camera than any bystander, so largest-area is a more
    stable choice than highest-confidence.
    """
    return max(boxes, key=lambda b: b.area) if boxes else None


# ---------------------------------------------------------------------------
# Face box calibration (YuNet box -> MiVOLO's YOLO face box convention)
# ---------------------------------------------------------------------------


def load_calibration(path: Optional[Path] = None) -> dict:
    path = path or (MODELS_DIR / "face_box_calibration.json")
    if not path.exists():
        return dict(IDENTITY_CALIBRATION)
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(box: FaceBox, calib: dict) -> FaceBox:
    """Map a YuNet box onto the box the MiVOLO detector would have produced.

    MiVOLO crops the raw bounding box its own YOLOv8x person+face detector emits
    (mivolo/structures.py :: crop_object -- no margin, no alignment). YuNet's
    boxes follow a different convention, so feeding them straight to MiVOLO would
    put the crop outside the distribution the model was trained on. The four
    constants here are measured by scripts/calibrate_face_box.py.
    """
    cx = box.cx + calib["shift_x"] * box.width
    cy = box.cy + calib["shift_y"] * box.height
    w = box.width * calib["scale_w"]
    h = box.height * calib["scale_h"]
    return FaceBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, box.score)


def clamp_box(box: FaceBox, width: int, height: int) -> FaceBox:
    """Clamp to image bounds.

    Mirrors PersonAndFaceResult.get_bbox_by_ind, which clamps to
    [0, w-1] / [0, h-1] and truncates to int before slicing.
    """
    x1 = min(max(int(box.x1), 0), width - 1)
    y1 = min(max(int(box.y1), 0), height - 1)
    x2 = min(max(int(box.x2), 0), width - 1)
    y2 = min(max(int(box.y2), 0), height - 1)
    return FaceBox(float(x1), float(y1), float(x2), float(y2), box.score)


def crop_face(image_bgr: np.ndarray, box: FaceBox) -> Optional[np.ndarray]:
    """mivolo/structures.py :: crop_object with cut_other_classes=[] -> plain slice."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# MiVOLO preprocessing
# ---------------------------------------------------------------------------


def letterbox(image: np.ndarray, new_shape: int, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Byte-for-byte port of mivolo/data/misc.py :: class_letterbox (scaleup=True)."""
    shape = image.shape[:2]  # h, w
    if shape[0] == new_shape and shape[1] == new_shape:
        return image

    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def normalise(image_bgr: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Port of mivolo/data/misc.py :: prepare_classification_images (non-None branch)."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = image / 255.0
    array = (array - np.asarray(mean)) / np.asarray(std)
    array = array.astype(np.float32).transpose(2, 0, 1)
    return np.ascontiguousarray(array)[None, ...]


# ---------------------------------------------------------------------------
# MiVOLO model
# ---------------------------------------------------------------------------


class MiVOLOOnnx:
    def __init__(self, model_path: Path, meta: dict) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.meta = meta
        self.input_name = meta.get("input_name", self.session.get_inputs()[0].name)
        self.output_name = meta.get("output_name", self.session.get_outputs()[0].name)
        self.input_size = int(meta["input_size"])
        self.in_chans = int(meta["in_chans"])
        self.mean = meta["mean"]
        self.std = meta["std"]

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        boxed = letterbox(face_bgr, self.input_size, tuple(self.meta.get("letterbox_pad_value", (0, 0, 0))))
        tensor = normalise(boxed, self.mean, self.std)
        if self.in_chans == 6:
            # Face+person checkpoint used face-only: MiVOLO fills a missing crop
            # with a zero image put through the same normalisation
            # (prepare_classification_images, img is None branch).
            zeros = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            for c in range(3):
                zeros[:, c] = (0.0 - self.mean[c]) / self.std[c]
            tensor = np.concatenate([tensor, zeros], axis=1)
        return tensor

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: tensor})[0]

    def decode(self, output: np.ndarray) -> Tuple[float, str, float]:
        """Port of mivolo/model/mi_volo.py :: fill_in_results."""
        row = output[0]
        meta = self.meta
        if meta["only_age"]:
            age_norm = float(row[0])
            gender, gender_score = "unknown", 0.0
        else:
            logits = row[list(meta["gender_logit_indices"])].astype(np.float64)
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            index = int(np.argmax(probs))
            gender = meta["gender_labels"][index]
            gender_score = float(probs[index])
            age_norm = float(row[int(meta["age_output_index"])])

        age = age_norm * (meta["max_age"] - meta["min_age"]) + meta["avg_age"]
        return round(age, 2), gender, gender_score


def tensor_stats(tensor: np.ndarray) -> dict:
    """A fingerprint used to diff the Python and Android input tensors."""
    flat = tensor.reshape(-1).astype(np.float64)
    return {
        "shape": list(tensor.shape),
        "sum": round(float(flat.sum()), 4),
        "mean": round(float(flat.mean()), 6),
        "min": round(float(flat.min()), 6),
        "max": round(float(flat.max()), 6),
        "first8": [round(float(v), 6) for v in flat[:8]],
    }


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def resolve_paths() -> Tuple[Path, Path, Path]:
    """Prefer models/, fall back to the packaged Android assets."""

    def pick(name: str) -> Path:
        for candidate in (MODELS_DIR / name, ASSETS_DIR / name):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{name} not found in {MODELS_DIR} or {ASSETS_DIR}.\n"
            "Run: python scripts/download_models.py && python scripts/export_model.py"
        )

    meta_path = pick("model_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return pick(meta["model_file"]), pick("face_detection_yunet_2023mar.onnx"), meta_path


class AgeGenderPipeline:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        detector_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        calibration: Optional[dict] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        if model_path is None or detector_path is None or meta_path is None:
            model_path, detector_path, meta_path = resolve_paths()
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.model = MiVOLOOnnx(Path(model_path), self.meta)
        self.detector = YuNetDetector(Path(detector_path), score_threshold=score_threshold)
        self.calibration = calibration if calibration is not None else load_calibration()

    def predict(self, image_bgr: np.ndarray) -> Tuple[Optional[Prediction], List[FaceBox]]:
        faces = self.detector.detect(image_bgr)
        primary = select_primary_face(faces)
        if primary is None:
            return None, faces

        calibrated = apply_calibration(primary, self.calibration)
        crop = crop_face(image_bgr, calibrated)
        if crop is None or crop.size == 0:
            return None, faces

        tensor = self.model.preprocess(crop)
        output = self.model.infer(tensor)
        age, gender, gender_score = self.model.decode(output)
        prediction = Prediction(
            age=age,
            gender=gender,
            gender_score=gender_score,
            box=calibrated,
            raw_output=output,
            input_tensor_stats=tensor_stats(tensor),
        )
        return prediction, faces
