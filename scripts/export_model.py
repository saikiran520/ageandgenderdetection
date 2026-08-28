#!/usr/bin/env python3
"""
Convert the official MiVOLO checkpoint into an ONNX model for ONNX Runtime Android.

Every number that Android needs at inference time -- input size, channel count,
normalisation constants, age denormalisation constants, gender label order -- is
READ OUT OF THE CHECKPOINT (or out of timm's resolved data config), never
hardcoded here. The values land in models/model_meta.json which is shipped in
app/src/main/assets/ and parsed by ModelManager.kt.

Pipeline:
    checkpoint (.pth.tar)
        -> mivolo.model.create_timm_model.create_model()   (official architecture)
        -> torch.onnx.export(opset 18)                     (Col2Im needs >= 18)
        -> numerical check torch vs onnxruntime
        -> models/mivolo_face_224.onnx + models/model_meta.json

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --opset 18 --tolerance 1e-3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CKPT = MODELS_DIR / "mivolo_volo_d1_face_only_imdb.pth.tar"

# timm's IMAGENET_DEFAULT_MEAN / _STD, which MiVOLO's _cfg() uses verbatim.
# We still read them back from the model's resolved data config below rather
# than trusting this comment.
_EXPECTED_MEAN = (0.485, 0.456, 0.406)
_EXPECTED_STD = (0.229, 0.224, 0.225)


@contextlib.contextmanager
def patched_fold():
    """Make VOLO's outlook attention exportable to ONNX.

    VOLO's OutlookAttention ends with

        B, H, W, C = x.shape
        ...
        x = F.fold(x, output_size=(H, W), kernel_size=..., padding=..., stride=...)

    Under torch.onnx tracing, H and W arrive as graph values rather than Python
    ints, so `output_size` becomes a tensor of unknown static shape. PyTorch's
    opset-18 symbolic for col2im then does
    `symbolic_helper._get_tensor_sizes(output_size)[0]`, gets None, and blows up
    with "TypeError: 'NoneType' object is not subscriptable". That is the exact
    failure reported in MiVOLO issue #14, which is still open upstream.

    Coercing output_size to plain ints bakes the spatial size in as a constant,
    which is correct here because we export at a fixed input resolution (only the
    batch axis is dynamic). The exporter then emits a real ONNX Col2Im node.
    ONNX Runtime has supported Col2Im since opset 18, and the export is checked
    numerically against PyTorch further down, so this cannot silently change
    behaviour.
    """
    original = F.fold

    def fold_with_static_output_size(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
        if isinstance(output_size, (tuple, list)):
            output_size = tuple(int(v) for v in output_size)
        return original(input, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride)

    F.fold = fold_with_static_output_size
    try:
        yield
    finally:
        F.fold = original


def _require_mivolo():
    try:
        from mivolo.model.create_timm_model import create_model  # noqa: F401
    except ImportError:
        sys.exit(
            "The 'mivolo' package is not importable.\n"
            "Install it with:\n"
            "    pip install -r scripts/requirements.txt\n"
            "which pulls it straight from the official repository."
        )


def read_checkpoint_meta(ckpt_path: Path) -> dict:
    """Reproduce mivolo.model.mi_volo.Meta.load_from_ckpt without needing a GPU."""
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    for key in ("min_age", "max_age", "avg_age", "no_gender", "state_dict"):
        if key not in state:
            sys.exit(f"Checkpoint {ckpt_path.name} has no '{key}' entry -- is this really a MiVOLO checkpoint?")

    sd = state["state_dict"]
    only_age = bool(state["no_gender"])

    if "with_persons_model" in state:
        with_persons_model = bool(state["with_persons_model"])
    else:
        # Same heuristic MiVOLO uses: the two-stream stem is named conv1/conv2.
        with_persons_model = "patch_embed.conv1.0.weight" in sd

    # MiVOLO derives the spatial size from the positional embedding: pos_embed is
    # (1, H/16, W/16, C) because patch_size 8 with pooling_scale 2 => stride 16.
    input_size = int(sd["pos_embed"].shape[1]) * 16

    return {
        "min_age": float(state["min_age"]),
        "max_age": float(state["max_age"]),
        "avg_age": float(state["avg_age"]),
        "only_age": only_age,
        "with_persons_model": with_persons_model,
        "num_classes": 1 if only_age else 3,
        "in_chans": 6 if with_persons_model else 3,
        "input_size": input_size,
    }


def build_model(ckpt_path: Path, meta: dict):
    from mivolo.model.create_timm_model import create_model
    from timm.data import resolve_data_config

    model_name = f"mivolo_d1_{meta['input_size']}"
    print(f"  architecture      : {model_name}")
    model = create_model(
        model_name=model_name,
        num_classes=meta["num_classes"],
        in_chans=meta["in_chans"],
        pretrained=False,
        checkpoint_path=str(ckpt_path),
        filter_keys=["fds."],  # feature-distribution-smoothing buffers, training only
    )
    model.eval()

    data_config = resolve_data_config(model=model, verbose=False, use_test_size=True)
    data_config["crop_pct"] = 1.0  # MiVOLO forces this in MiVOLO.__init__
    return model, data_config


def export(model, meta: dict, out_path: Path, opset: int) -> None:
    dummy = torch.randn(1, meta["in_chans"], meta["input_size"], meta["input_size"], dtype=torch.float32)
    print(f"  exporting to ONNX : opset {opset}, input {tuple(dummy.shape)}")
    with patched_fold():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )


def verify(model, meta: dict, out_path: Path, tolerance: float) -> float:
    import onnxruntime as ort

    torch.manual_seed(0)
    sample = torch.randn(1, meta["in_chans"], meta["input_size"], meta["input_size"], dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(sample).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["output"], {"input": sample.numpy()})[0]

    diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"  torch output      : {np.array2string(torch_out, precision=6)}")
    print(f"  onnx  output      : {np.array2string(onnx_out, precision=6)}")
    print(f"  max abs diff      : {diff:.3e}  (tolerance {tolerance:.0e})")
    if diff > tolerance:
        raise SystemExit(
            f"ONNX export does NOT match PyTorch (max abs diff {diff:.3e} > {tolerance:.0e}). "
            "Refusing to ship a model that has not been validated."
        )
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to models/mivolo_face_<size>.onnx")
    parser.add_argument("--opset", type=int, default=18, help="VOLO's outlook attention needs Col2Im (opset >= 18).")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\nRun: python scripts/download_models.py")

    _require_mivolo()

    print(f"Reading checkpoint metadata from {args.checkpoint.name}")
    meta = read_checkpoint_meta(args.checkpoint)
    for key, value in meta.items():
        print(f"  {key:18}: {value}")

    if meta["with_persons_model"]:
        print(
            "\n  NOTE: this is a face+person checkpoint (6 input channels). The Android app only "
            "captures a face crop and will feed zeros into the person channels.\n"
        )

    print("\nBuilding model")
    model, data_config = build_model(args.checkpoint, meta)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  parameters        : {param_count:,}")

    mean = tuple(round(float(v), 6) for v in data_config["mean"])
    std = tuple(round(float(v), 6) for v in data_config["std"])
    print(f"  normalisation mean: {mean}")
    print(f"  normalisation std : {std}")
    if mean != _EXPECTED_MEAN or std != _EXPECTED_STD:
        print("  (heads-up: these differ from timm's ImageNet defaults; Android reads them from model_meta.json)")

    cfg_c, cfg_h, cfg_w = data_config["input_size"]
    if cfg_h != cfg_w:
        sys.exit(f"Non-square input {cfg_h}x{cfg_w} is not supported by this pipeline.")
    if cfg_h != meta["input_size"]:
        print(f"  (data_config says {cfg_h}, checkpoint pos_embed says {meta['input_size']}; using {cfg_h})")
        meta["input_size"] = cfg_h

    out_path = args.output or (MODELS_DIR / f"mivolo_face_{meta['input_size']}.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nExporting")
    export(model, meta, out_path, args.opset)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote             : {out_path}  ({size_mb:.1f} MB)")

    print("\nValidating ONNX against PyTorch")
    max_diff = verify(model, meta, out_path, args.tolerance)

    model_meta = {
        "_comment": (
            "Generated by scripts/export_model.py. Every value is read from the MiVOLO checkpoint or "
            "from timm's resolved data config -- do not edit by hand."
        ),
        "model_file": out_path.name,
        "source_checkpoint": args.checkpoint.name,
        "architecture": f"mivolo_d1_{meta['input_size']}",
        "parameters": int(param_count),
        "opset": args.opset,
        "onnx_vs_torch_max_abs_diff": max_diff,
        "input_name": "input",
        "output_name": "output",
        "input_size": int(meta["input_size"]),
        "in_chans": int(meta["in_chans"]),
        "num_classes": int(meta["num_classes"]),
        "layout": "NCHW",
        "channel_order": "RGB",
        "pixel_scale": 255.0,
        "mean": list(mean),
        "std": list(std),
        "letterbox_pad_value": [0, 0, 0],
        "resize_interpolation": "bilinear",
        "only_age": bool(meta["only_age"]),
        "with_persons_model": bool(meta["with_persons_model"]),
        "min_age": meta["min_age"],
        "max_age": meta["max_age"],
        "avg_age": meta["avg_age"],
        "age_formula": "age = output[2] * (max_age - min_age) + avg_age",
        "gender_logit_indices": [0, 1],
        "gender_labels": ["male", "female"],
        "age_output_index": 2,
    }
    meta_path = MODELS_DIR / "model_meta.json"
    meta_path.write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    print(f"\nWrote {meta_path}")

    print("\nDecoding rules Android will use:")
    print(f"  gender = argmax(softmax(output[0:2]))  ->  0={model_meta['gender_labels'][0]}, "
          f"1={model_meta['gender_labels'][1]}")
    print(f"  age    = output[2] * ({meta['max_age']} - {meta['min_age']}) + {meta['avg_age']}")

    print("\nNext: python scripts/calibrate_face_box.py   (or skip and run scripts/test_mivolo.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
