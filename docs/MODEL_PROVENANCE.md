# Model provenance

Exactly which files this POC uses, where each one comes from, what licence it
carries, and how it becomes part of the APK. Nothing here was invented; every
URL is taken from the official MiVOLO repository or the OpenCV Zoo.

---

## 1. Age + gender model

| | |
|---|---|
| **What** | MiVOLO `volo_d1`, **face-only**, age + gender |
| **Trained on** | IMDB-cleaned |
| **Published metrics** | Age MAE 4.22 · CS@5 68.68 · Gender accuracy 99.38 % |
| **Source** | The model table in the [official MiVOLO README](https://github.com/WildChlamydia/MiVOLO#mivolo-pretrained-models), row *"volo_d1 / face_only, age, gender / IMDB-cleaned"* |
| **Direct link** | https://drive.google.com/file/d/1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP/view |
| **Licence** | MiVOLO repository licence — https://github.com/WildChlamydia/MiVOLO/blob/main/license |
| **Downloaded to** | `models/mivolo_volo_d1_face_only_imdb.pth.tar` (103 498 750 bytes) |
| **Downloaded by** | `scripts/download_models.py` |

### Why this checkpoint

MiVOLO publishes both *face-only* (3-channel) and *face+body* (6-channel)
checkpoints. This POC captures a single photograph of a face, so the face-only
variant is the right configuration: it needs no person detector, no person crop
and no person-body association step, and at 224×224 it costs roughly a sixth of
the compute of `mivolov2_d1_384x384`.

`scripts/export_model.py` also supports the 6-channel checkpoints — it reads the
channel count out of the checkpoint and, when a face+person model is used
face-only, fills the person channels the way MiVOLO itself does (a zero image put
through the same normalisation, per `prepare_classification_images`).

### Values read out of the checkpoint

These are **read, not assumed**. `export_model.py` pulls them from the checkpoint
and writes them to `models/model_meta.json`, which is packaged into the APK and
parsed by `ModelManager.kt`. Changing checkpoint needs no Kotlin change.

```
min_age            : 1.0
max_age            : 95.0
avg_age            : 48.0
no_gender          : False        -> num_classes = 3
with_persons_model : False        -> in_chans    = 3
input_size         : 224          (from state_dict["pos_embed"].shape[1] * 16)
parameters         : 25,864,350
normalisation mean : (0.485, 0.456, 0.406)     timm IMAGENET_DEFAULT_MEAN
normalisation std  : (0.229, 0.224, 0.225)     timm IMAGENET_DEFAULT_STD
```

Decoding, ported from `mivolo/model/mi_volo.py :: fill_in_results`:

```
gender = argmax(softmax(output[0:2]))     index 0 = male, 1 = female
age    = output[2] * (95.0 - 1.0) + 48.0
```

---

## 2. Face detector (shipped)

| | |
|---|---|
| **What** | YuNet, `face_detection_yunet_2023mar.onnx` |
| **Source** | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| **Licence** | MIT (OpenCV Zoo / libfacedetection) |
| **Size** | 232 589 bytes |
| **Downloaded to** | `models/face_detection_yunet_2023mar.onnx` |

### Why not MiVOLO's own detector

MiVOLO ships `yolov8x_person_face.pt`, a **68.1 M parameter** YOLOv8x. As ONNX
that is roughly 270 MB and takes seconds per frame on a phone CPU — it would
dominate both the APK and the latency budget for a POC that analyses one still
photograph. It is also AGPL-3.0, which is a licensing consideration for an app.

YuNet is 232 KB, MIT licensed, and runs in the same ONNX Runtime session pool as
MiVOLO, which means **the Python reference script and the Android app execute the
identical detector graph**. That is what makes the Python-vs-Android comparison
in the README a real check rather than a coincidence.

### Making YuNet's boxes match MiVOLO's

MiVOLO crops the *raw* box its own detector emits — `mivolo/structures.py ::
crop_object` slices `full_image[y1:y2, x1:x2]` with no margin and no alignment.
The age head is therefore sensitive to how tight that box is, and a different
detector's box convention would put the crop outside the model's training
distribution.

Rather than guessing a margin, `scripts/calibrate_face_box.py` runs **both**
detectors over the same photographs, matches boxes by IoU, and measures the
median transform. Measured over 42 public-domain portraits, 41 matched face
pairs:

```
scale_w : 0.9777    IQR [0.9595, 0.9917]
scale_h : 0.8776    IQR [0.8632, 0.9254]
shift_x : +0.0063   IQR [-0.0045, 0.0153]
shift_y : +0.0470   IQR [0.0368, 0.0693]

mean IoU against the official detector: 0.8453  ->  0.9226
```

The transform lands in `models/face_box_calibration.json`, is packaged into the
APK, and is applied identically by `mivolo_pipeline.py` (Python) and
`FaceDetector.kt` (Kotlin).

---

## 3. Reference detector (never shipped)

| | |
|---|---|
| **What** | `yolov8x_person_face.pt`, MiVOLO's official person+face detector |
| **Source** | https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector |
| **Licence** | AGPL-3.0 |
| **Size** | 136 716 488 bytes |
| **Used by** | `scripts/calibrate_face_box.py`, `scripts/test_mivolo.py --reference` |

Downloaded only with `python scripts/download_models.py --with-detector-reference`.
It exists purely so the shipped pipeline can be checked against the original one.
It is **not** copied into `app/src/main/assets/` and is not present in the APK.

---

## 4. Test images

Public-domain photographs from Wikimedia Commons, category *"Official portraits
of members of the 118th United States Congress"* — works of the US federal
government. Fetched by `scripts/fetch_testdata.py` into `testdata/`, with
per-file provenance recorded in `testdata/SOURCES.json`.

Only one of them is packaged, as `app/src/main/assets/test.jpg`, to drive the
in-app self-test that proves Python and Android agree.

---

## 5. Export path: why ONNX

The MiVOLO README recommends TorchScript and warns that ONNX export "is not
advisable due to the poor performance of the resulting model with batch
processing". That warning is about **batched** throughput on a server. This POC
runs batch size 1 on a phone, where the concern does not apply, and ONNX Runtime
is the runtime with a supported, current Android distribution — PyTorch Mobile is
deprecated in favour of ExecuTorch, and exporting VOLO's outlook attention
through ExecuTorch is considerably more fragile than through ONNX.

### The `col2im` problem, and the fix

A naive `torch.onnx.export` of MiVOLO fails:

```
File "torch/onnx/symbolic_opset18.py", line 75, in col2im
    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
TypeError: 'NoneType' object is not subscriptable
(Occurred when translating col2im).
```

This is exactly the failure reported in
[MiVOLO issue #14](https://github.com/WildChlamydia/MiVOLO/issues/14), which is
still open upstream with no workaround given.

**Cause.** VOLO's `OutlookAttention.forward` ends with

```python
B, H, W, C = x.shape
...
x = F.fold(x, output_size=(H, W), kernel_size=..., padding=..., stride=...)
```

Under tracing, `H` and `W` arrive as graph values rather than Python ints, so
`output_size` becomes a tensor of unknown static shape and PyTorch's opset-18
`col2im` symbolic cannot read its size.

**Fix.** `scripts/export_model.py` wraps the export in a context manager that
coerces `output_size` to plain ints. That is correct here because the export
pins the input resolution and leaves only the batch axis dynamic. The exporter
then emits a genuine ONNX `Col2Im` node, which ONNX Runtime has supported since
opset 18.

The fix is five lines and touches nothing else:

```python
def fold_with_static_output_size(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if isinstance(output_size, (tuple, list)):
        output_size = tuple(int(v) for v in output_size)
    return original(input, output_size, kernel_size, dilation=dilation, padding=padding, stride=stride)
```

**Verification.** The export is not trusted on faith — `export_model.py` runs the
PyTorch model and the ONNX model on the same input and refuses to write
`model_meta.json` if they disagree:

```
torch output : [[ 0.018213 -0.072361 -0.116727]]
onnx  output : [[ 0.018213 -0.072361 -0.116727]]
max abs diff : 5.215e-08   (tolerance 1e-03)
```

---

## 6. What ends up in the APK

```
app/src/main/assets/
├── mivolo_face_224.onnx                98.9 MB   the age+gender model
├── face_detection_yunet_2023mar.onnx  227.1 KB   the face detector
├── model_meta.json                      1.1 KB   input size, normalisation, age decoding
├── face_box_calibration.json             917 B   YuNet -> MiVOLO box transform
└── test.jpg                            258.7 KB  self-test image
```

Plus ONNX Runtime's native libraries for `arm64-v8a` and `x86_64`.

`AndroidManifest.xml` requests **`CAMERA` only**. There is deliberately no
`INTERNET` permission, so the offline guarantee is enforced by the operating
system rather than merely asserted in documentation.
