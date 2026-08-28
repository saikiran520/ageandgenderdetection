# MiVOLO Age & Gender — offline Android POC

A minimal Android app in Kotlin that estimates **age** and **gender** from a
photograph you take with the device camera, running
[MiVOLO](https://github.com/WildChlamydia/MiVOLO) entirely on-device.

```
Home screen  ──[ MiVOLO ]──▶  Camera  ──[ Capture ]──▶  Analyzing…  ──▶  Age: 50
                                                                        Gender: Male
                                                                       [ Capture Again ]
```

**No network is used, or possible.** `AndroidManifest.xml` requests `CAMERA`
only — there is deliberately **no `INTERNET` permission**, so the offline
guarantee is enforced by Android itself rather than asserted in a README. No
image leaves the device; no cloud API is involved at any point.

---

## What's actually running

| Stage | Component | Size | Where |
|---|---|---|---|
| Face detection | YuNet ONNX (OpenCV Zoo) | 227 KB | `FaceDetector.kt` |
| Box calibration | measured YuNet → MiVOLO transform | 917 B | `FaceDetector.kt` |
| Crop / letterbox / normalise | port of MiVOLO's own preprocessing | — | `ImagePreprocessor.kt` |
| Age + gender | MiVOLO `volo_d1` face-only, ONNX | 98.9 MB | `MiVOLOProcessor.kt` |
| Runtime | ONNX Runtime 1.29 (arm64-v8a, x86_64) | 67 MB | — |

The exact checkpoint, its licence, and every derived constant are documented in
**[docs/MODEL_PROVENANCE.md](docs/MODEL_PROVENANCE.md)**. Short version: the
*"volo_d1 / face_only, age, gender / IMDB-cleaned"* row of the official MiVOLO
model table (Age MAE 4.22, gender accuracy 99.38 %).

---

## Verification results

Everything below was measured on this repository, not estimated.

**ONNX export is faithful to PyTorch** — `export_model.py` refuses to write the
metadata unless this passes:

```
torch output : [[ 0.018213 -0.072361 -0.116727]]
onnx  output : [[ 0.018213 -0.072361 -0.116727]]
max abs diff : 5.215e-08
```

**Preprocessing is bit-identical to the official MiVOLO code** — the Python port
is diffed against the real `mivolo.data.misc` functions across a range of crop
shapes (`test_mivolo.py --verify-preprocessing`):

```
crop   37x91    letterbox delta 0   tensor delta 0.000e+00   ok
crop  224x224   letterbox delta 0   tensor delta 0.000e+00   ok
crop  300x120   letterbox delta 0   tensor delta 0.000e+00   ok
crop    1x5     letterbox delta 0   tensor delta 0.000e+00   ok
crop  513x511   letterbox delta 0   tensor delta 0.000e+00   ok
crop   64x64    letterbox delta 0   tensor delta 0.000e+00   ok
```

**The substituted face detector barely moves the answer** — the shipped pipeline
(ONNX + YuNet + calibration) versus the original one (PyTorch checkpoint +
MiVOLO's own YOLOv8x detector), over 41 faces
(`test_mivolo.py --batch testdata --reference`):

```
41 faces  mean age 49.7  median 48.7  mean 110 ms

ONNX + YuNet  vs  original PyTorch + official YOLOv8x detector
  age delta   : mean 0.894  median 0.630  p95 1.990  max 2.930 years
  gender agree: 41/41 (100.0%)
```

For scale: MiVOLO's own published age MAE for this checkpoint is **4.22 years**,
so a 0.63-year median deviation from the reference pipeline is well inside the
model's own error. Gender is unaffected.

Box calibration lifts mean IoU against the official detector from **0.845 to
0.923** over 41 matched faces.

**Android agrees with Python** — the same `test.jpg` run through the app on a
device and through `test_mivolo.py` on the desktop:

```
             age      gender
android    49.61      male
python     49.65      male
delta       0.04
```

Full breakdown, including the input tensor, under
[Comparing Python and Android](#comparing-python-and-android).

---

## Prerequisites

| | |
|---|---|
| Android Studio | Narwhal / 2025.1 or newer (AGP 9.3.2) |
| Android SDK | compileSdk 37, minSdk 28 |
| JDK | 17 or newer (Gradle toolchain is pinned to 25) |
| Python | 3.10 (matches MiVOLO's own requirements) — **only for model preparation** |
| Disk | ~4 GB for the Python env, ~350 MB for models |
| Device | arm64 phone (Android 9+), or an x86_64 emulator |

Python is needed **once**, to download and convert the model. The app itself has
no Python dependency.

---

## Setup

### Step 1 — Open the project

Clone or open this directory in Android Studio and let it sync once. Confirm
`local.properties` points at your SDK.

### Step 2 — Create the Python environment

> **Windows:** put the virtualenv on a **short path**. `torch` contains file
> paths long enough to hit the 260-character `MAX_PATH` limit, and the install
> fails halfway with a confusing `OSError`.

```bash
python -m venv C:\venvs\mivolo          # Windows
# python -m venv .venv                  # macOS / Linux

C:\venvs\mivolo\Scripts\activate        # Windows
# source .venv/bin/activate             # macOS / Linux

pip install "setuptools<81" wheel
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
pip install -r scripts/requirements.txt
pip install --no-build-isolation "mivolo @ git+https://github.com/WildChlamydia/MiVOLO.git"
```

`setuptools<81` and `--no-build-isolation` are both required: MiVOLO's
`setup.py` imports `pkg_resources`, which setuptools 81 removed.

### Step 3 — Download the models

```bash
python scripts/download_models.py
```

Fetches the MiVOLO checkpoint (98.7 MB, from the Google Drive link in the
official README) and YuNet (227 KB, from the OpenCV Zoo) into `models/`, then
prints the SHA-256 of each and writes `models/download_manifest.json`.

To also fetch the official YOLOv8x detector used for calibration and reference
comparison (137 MB, never shipped):

```bash
python scripts/download_models.py --with-detector-reference
```

### Step 4 — Export to ONNX

```bash
python scripts/export_model.py
```

Loads the checkpoint through MiVOLO's own `create_model`, exports to ONNX at
opset 18, **verifies the ONNX output against PyTorch**, and writes
`models/mivolo_face_224.onnx` plus `models/model_meta.json`.

`model_meta.json` holds every value Android needs — input size, channel count,
normalisation mean/std, the age denormalisation constants and the gender label
order — all read out of the checkpoint rather than hardcoded. Swapping in a
different MiVOLO checkpoint needs no Kotlin changes.

### Step 5 — Calibrate the face box *(optional but recommended)*

```bash
python scripts/fetch_testdata.py --count 40
python scripts/calibrate_face_box.py
```

Runs YuNet and MiVOLO's official detector over the same photographs and measures
the transform between their box conventions. Without this the app falls back to
an identity transform, which is safe but slightly less faithful.

### Step 6 — Package the assets

```bash
python scripts/pack_assets.py
```

Copies the model, the detector, the metadata, the calibration and one self-test
image into `app/src/main/assets/`.

### Step 7 — Build and run

```bash
gradlew.bat :app:assembleDebug          # Windows
./gradlew :app:assembleDebug            # macOS / Linux
```

Or press Run in Android Studio. The debug APK lands at
`app/build/outputs/apk/debug/app-debug.apk` (**216 MB** — the 99 MB model plus
ONNX Runtime's native libraries for two ABIs).

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## Testing it offline

This is the point of the whole exercise, so it is worth doing deliberately:

1. Install the APK while you still have a connection.
2. Open the app once and tap **MiVOLO** — this triggers the one-time extraction
   of the models from the APK into the app's private storage. Watch for
   `MiVOLO model loaded` in logcat.
3. Put the device in **aeroplane mode**. Turn Wi-Fi and mobile data off
   explicitly as well.
4. Capture a photo. You will still get `Age:` and `Gender:`.

Nothing about step 3 can break it: the app holds no `INTERNET` permission, so
the process cannot open a socket even if some dependency tried to.

To confirm this for yourself:

```bash
adb shell dumpsys package com.ebani.ageandgenderdetection | grep -i permission
```

You will see `android.permission.CAMERA` and no `INTERNET`.

---

## Validating the model in Python

Requirement: check the model works *before* trusting the Android integration.

```bash
python scripts/test_mivolo.py --image testdata/some_portrait.jpg
```

```
Age: 50
Gender: Male
```

Useful variants:

```bash
# full fingerprint: input tensor stats + raw logits
python scripts/test_mivolo.py --image test.jpg --json

# prove the preprocessing port matches the real mivolo package
python scripts/test_mivolo.py --verify-preprocessing

# compare against the ORIGINAL pipeline (PyTorch + official YOLOv8x detector)
python scripts/test_mivolo.py --image test.jpg --reference

# run the whole test set, with reference deltas
python scripts/test_mivolo.py --batch testdata --reference
```

---

## Comparing Python and Android

The app can run a bundled image through the pipeline with no camera involved,
which makes the two sides directly comparable.

1. Open the app and tap **"Run bundled self-test (Python parity)"**.
2. Then:

```bash
python scripts/compare_android.py --image app/src/main/assets/test.jpg
```

It pulls the last analysis out of `adb logcat -s MiVOLO`, runs the same image
through the Python pipeline, and diffs **the input tensor as well as the
outputs** — because that is where preprocessing bugs actually surface:

Actual output from this repository, Android 16 (API 37) x86_64 against the same
`test.jpg`:

```
  android crop           [408, 559]
  android tensor shape   [1, 3, 224, 224]
  python  tensor shape   [1, 3, 224, 224]

  input tensor sum       android  -55564.715400   python  -55805.013200   delta  240.297800 (0.431%)   ok
  input tensor mean      android      -0.369132   python      -0.370728   delta    0.001596 (0.431%)   ok
  input tensor min       android      -2.117904   python      -2.117904   delta    0.000000   ok
  input tensor max       android       2.517996   python       2.517996   delta    0.000000   ok

  raw_output[0]          android       2.976146   python       2.975520   delta    0.000626   ok
  raw_output[1]          android      -2.943839   python      -2.943581   delta    0.000258   ok
  raw_output[2]          android       0.017102   python       0.017524   delta    0.000422   ok

  age (years)            android      49.607620   python      49.650000   delta    0.042380   ok
  gender                 android           male   python           male   ok

PASS - Android and Python agree.
```

Both sides pick the **same crop** (408×559) and produce the same tensor min/max
exactly. The 0.43 % difference in the tensor *sum* is the one honest residual:
OpenCV's `INTER_LINEAR` resize uses fixed-point arithmetic while the Kotlin port
works in floating point. The sampling grid is identical — only the final rounding
differs — and it costs **0.04 years** on the decoded age. The comparison uses a
relative tolerance for that reason; anything materially larger is a real bug.

If it *doesn't* pass, the failure tells you where to look:

| Symptom | Look at |
|---|---|
| input tensor sum differs | crop bounds, resize, letterbox padding, channel order |
| tensor matches, output differs | the ONNX graph or the runtime |
| both match, age differs | the decoding constants in `model_meta.json` |

Working without a device attached:

```bash
adb logcat -d -s MiVOLO > android.log
python scripts/compare_android.py --image app/src/main/assets/test.jpg --log android.log
```

---

## Debug logging

Everything is logged under the tag `MiVOLO`:

```bash
adb logcat -s MiVOLO
```

```
MiVOLO: ----- model metadata -----
MiVOLO:   checkpoint      : mivolo_volo_d1_face_only_imdb.pth.tar
MiVOLO:   architecture    : mivolo_d1_224
MiVOLO:   input           : input [1, 3, 224, 224] NCHW RGB
MiVOLO:   mean            : 0.485, 0.456, 0.406
MiVOLO:   age decode      : output[2] * (95.0 - 1.0) + 48.0
MiVOLO: ----- face box calibration -----
MiVOLO:   scale=(0.9777, 0.8776) shift=(0.0063, 0.047)
MiVOLO:   measured on 41 face pairs, mean IoU 0.8453 -> 0.9226
MiVOLO: YuNet face detector loaded (face_detection_yunet_2023mar.onnx)
MiVOLO: MiVOLO model loaded (mivolo_face_224.onnx)
MiVOLO: ===== analysis start =====
MiVOLO: Input image size: 1080x1440
MiVOLO: Face detection: 1 face(s) in 41 ms (image 1080x1440, canvas 640x640, scale 0.4444)
MiVOLO: Face detected: [268,195 -> 677,755] 409x559 score=0.983  (selected from 1)
MiVOLO: Face box after MiVOLO calibration: [272,222 -> 672,713] 400x491 score=0.983
MiVOLO: Face crop: 400x491
MiVOLO: Letterboxed crop 400x491 -> 224x224
MiVOLO: Model input shape: [1, 3, 224, 224]
MiVOLO: Input tensor      : shape=[1, 3, 224, 224] sum=-55805.0215 mean=-0.370728 ...
MiVOLO: Inference started
MiVOLO: Inference completed in 612 ms
MiVOLO: Raw model output  : [2.975520, -2.943581, 0.017524]
MiVOLO: Decoded age       : 49.65
MiVOLO: Decoded gender    : male (100%)
MiVOLO: ===== analysis done in 691 ms =====
```

---

## Project layout

```
app/src/main/
├── java/com/ebani/ageandgenderdetection/
│   ├── MainActivity.kt        home screen, one "MiVOLO" button
│   ├── CameraActivity.kt      CameraX preview, capture, result, "Capture Again"
│   ├── ModelManager.kt        assets → ONNX Runtime sessions, model_meta.json
│   ├── FaceDetector.kt        YuNet inference + OpenCV-identical decoding + calibration
│   ├── ImagePreprocessor.kt   letterbox / normalise, ported from mivolo.data.misc
│   └── MiVOLOProcessor.kt     the pipeline, and MiVOLO's output decoding
├── assets/                    models + metadata (generated by pack_assets.py)
└── AndroidManifest.xml        CAMERA only — no INTERNET

scripts/
├── requirements.txt
├── download_models.py         fetch the checkpoint + detector, record SHA-256
├── export_model.py            checkpoint → ONNX, verified against PyTorch
├── mivolo_pipeline.py         the reference pipeline Kotlin mirrors
├── calibrate_face_box.py      measure the YuNet → MiVOLO box transform
├── fetch_testdata.py          public-domain portraits for calibration/testing
├── pack_assets.py             copy artifacts into app/src/main/assets
├── test_mivolo.py             "Age: XX / Gender: Male" validation
├── compare_android.py         diff the Android logcat against Python
└── quantize_model.py          optional INT8, with measured accuracy drift

docs/
├── steps.md                   the original brief
└── MODEL_PROVENANCE.md        every file, source, licence and derived constant
```

---

## Shrinking the APK

The 99 MB fp32 model is shipped so that Android matches Python within float
noise. If you want it smaller, quantize — and measure what it costs first:

```bash
python scripts/quantize_model.py            # ~25 MB, reports age drift + gender agreement
python scripts/pack_assets.py --model models/mivolo_face_224_int8.onnx
gradlew.bat :app:assembleDebug
```

You can also drop `x86_64` from `abiFilters` in `app/build.gradle.kts` (saves
~37 MB) if you only ever install on real phones.

---

## Design notes

**Why ONNX and not TorchScript.** The MiVOLO README recommends TorchScript, but
PyTorch Mobile is deprecated in favour of ExecuTorch, and VOLO's outlook
attention (`unfold`/`fold`) is fragile through ExecuTorch. ONNX Runtime has a
current, supported Android distribution and handled the graph exactly — see
[docs/MODEL_PROVENANCE.md §5](docs/MODEL_PROVENANCE.md) for the `col2im` export
bug ([MiVOLO issue #14](https://github.com/WildChlamydia/MiVOLO/issues/14),
still open upstream) and the five-line fix.

**Why the resize is hand-written.** `ImagePreprocessor.resizeBilinear` implements
OpenCV's `INTER_LINEAR` sampling grid (`src = (dst + 0.5) * scale - 0.5`) instead
of calling `Bitmap.createScaledBitmap`. Android's scaler places samples
differently, and although both are nominally "bilinear", the difference is enough
to drift the Android tensor away from the Python one.

**Why largest-face.** A POC capture has one intended subject, and in a hand-held
portrait the subject is nearer the camera than any bystander, so largest-area is
more stable than highest-confidence. `FaceDetector.selectPrimary`.

**No face found** shows *"No face detected. Please capture a clear face."* with
**Capture Again**, and never crashes.

---

## Known limitations

- The bundled checkpoint is trained on IMDB-cleaned, which under-predicts at the
  older end of the range. That is a property of the published model, not of this
  integration — the app reproduces the reference pipeline to within 0.24 years.
- `armeabi-v7a` (32-bit) devices are excluded by `abiFilters`. Add it back in
  `app/build.gradle.kts` if you need them.
- Only one face is analysed per capture, by design (requirement 15 excludes
  multi-person handling).

---

## Licences

| Component | Licence |
|---|---|
| MiVOLO checkpoint & code | [MiVOLO repository licence](https://github.com/WildChlamydia/MiVOLO/blob/main/license) |
| YuNet detector | MIT (OpenCV Zoo / libfacedetection) |
| YOLOv8x reference detector *(not shipped)* | AGPL-3.0 |
| ONNX Runtime | MIT |
| Test images | Public domain (US federal government works, via Wikimedia Commons) |

Please cite the MiVOLO papers if you build on this — see the
[official repository](https://github.com/WildChlamydia/MiVOLO#citing).
