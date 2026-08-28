I want you to build a **simple Proof of Concept Android application entirely in Kotlin** using the MiVOLO open-source project:

https://github.com/WildChlamydia/MiVOLO

## Goal

Create a very simple Android POC that does only this:

```text
Home Screen
    ↓
One button: "MiVOLO"
    ↓
User taps button
    ↓
Open Camera Screen
    ↓
User captures a photo
    ↓
Analyze the captured image locally using MiVOLO
    ↓
Display only:

Age: XX
Gender: Male/Female
```

This is only a POC. Do not add unnecessary features, authentication, database, API, cloud backend, analytics, or complex UI.

---

# Critical Requirements

## 1. Entire app must be Kotlin

Build the Android application entirely using Kotlin.

Use a standard modern Android project.

Preferred stack:

* Kotlin
* Android Studio compatible
* CameraX for camera capture
* XML layouts or Jetpack Compose; choose whichever is simpler and more reliable for this POC
* MVVM is optional; do not overengineer

The app should compile and run successfully.

---

# 2. MiVOLO must be used for Age + Gender

Use the official MiVOLO repository:

https://github.com/WildChlamydia/MiVOLO

The purpose is specifically:

* Age estimation
* Gender detection

Do NOT build face recognition or identity recognition.

The expected application flow is:

```text
Captured Image
      ↓
Detect face/person as required by MiVOLO
      ↓
Prepare the image exactly according to MiVOLO requirements
      ↓
Run MiVOLO
      ↓
Extract:
    - Estimated Age
    - Gender
      ↓
Display result in Android UI
```

Prefer the MiVOLO model/configuration that is appropriate for **face-only age and gender estimation**, if supported by the official repository.

Before implementing anything, inspect the official MiVOLO repository structure, documentation, model architecture, checkpoints, preprocessing, inference code, and output decoding.

Do not invent model filenames, tensor shapes, output values, preprocessing, or gender labels.

---

# 3. Must work completely offline during usage

This is extremely important.

After the required model files and runtime dependencies are included in the Android application, the following must work without:

* Wi-Fi
* Mobile data
* Internet connection
* Cloud API
* Backend server

The inference pipeline must be:

```text
Camera
   ↓
Captured Image
   ↓
Local Processing
   ↓
Local MiVOLO Model
   ↓
Age + Gender Result
```

No image should be uploaded anywhere.

No cloud AI API should be used.

No OpenAI API, Gemini API, Claude API, AWS API, Azure API, or any other online AI service should be used for inference.

---

# 4. Automatically obtain required MiVOLO files

During project setup/build preparation, automatically download or provide scripts to download all required open-source MiVOLO files from their official sources.

Do not expect me to manually search for model files.

Create a setup/download mechanism that obtains the required files.

For example, if necessary:

```text
scripts/
    download_models.sh
    download_models.py
```

or Gradle tasks if appropriate.

The implementation should clearly document:

1. Which model/checkpoint is being downloaded.
2. Where it comes from.
3. Its license/source.
4. Where it is stored.
5. How it is converted/exported for Android.
6. How it gets packaged into the Android app.

Only use official or clearly referenced sources from the MiVOLO project.

If a pretrained model requires downloading from Hugging Face, GitHub Releases, Google Drive, or another source referenced by the official MiVOLO repository, automate that process where possible.

---

# 5. Important: solve Android deployment properly

MiVOLO may not be directly usable as a standard Android model file.

Therefore, investigate the official repository and choose the most reliable deployment path.

Possible options include:

### Option A — TorchScript

```text
MiVOLO checkpoint
      ↓
Load official architecture
      ↓
Export correctly
      ↓
TorchScript model
      ↓
Android-compatible local runtime
```

### Option B — ONNX

Only use ONNX if the actual MiVOLO architecture can be exported and validated correctly.

```text
MiVOLO
    ↓
ONNX
    ↓
ONNX Runtime Android
```

### Option C — ExecuTorch

If MiVOLO can be properly exported to ExecuTorch:

```text
MiVOLO
    ↓
ExecuTorch .pte
    ↓
ExecuTorch Android
```

Choose the approach that is **actually compatible with the current MiVOLO architecture and Android**, not the one that is easiest to write theoretically.

The final Android app must successfully perform local inference.

---

# 6. Validate model before Android integration

Before integrating into Kotlin, create a Python validation script.

The script should:

```text
Input:
test.jpg

Output:
Age: XX
Gender: Male/Female
```

Example:

```bash
python test_mivolo.py --image test.jpg
```

This script must use the exact same:

* Model
* Preprocessing
* Input shape
* Normalization
* Output decoding

that will eventually be used in Android.

Before moving to Android, verify that inference works correctly.

---

# 7. Compare Python output with Android output

This is important.

After Android integration, the same test image should be tested through:

```text
Python MiVOLO
       vs
Android MiVOLO
```

The results should be reasonably consistent.

If they differ significantly, debug:

* RGB/BGR order
* Image resize
* Face crop
* Normalization
* Tensor layout
* Float precision
* Output decoding

Do not assume Android preprocessing is correct without validation.

---

# 8. Android UI requirements

Keep the UI extremely simple.

## Screen 1 — MainActivity

Only display:

```text
MiVOLO Age & Gender POC

[ MiVOLO ]
```

The button text must be:

```text
MiVOLO
```

When the user taps it, navigate to the camera screen.

---

## Screen 2 — Camera Screen

Show:

```text
Camera Preview


[ Capture ]
```

When the user taps Capture:

1. Capture one image.
2. Stop or freeze the preview if needed.
3. Show a loading/progress state:

```text
Analyzing...
```

4. Run local age + gender inference.
5. Display below the captured image:

```text
Age: 27

Gender: Male
```

Also provide:

```text
[ Capture Again ]
```

No additional analytics are needed.

---

# 9. Handle no-face situations

If no usable face is detected, display:

```text
No face detected.
Please capture a clear face.
```

Then allow:

```text
[ Capture Again ]
```

Do not crash.

---

# 10. Face detection

MiVOLO may require a face detector or person detector as part of its pipeline.

Inspect the official repository and use whatever detection pipeline is required for correct MiVOLO inference.

If an additional face detector is needed, prefer a lightweight local/open-source solution that can run fully offline.

For example, use a local model/runtime only if necessary.

Do not use cloud face detection.

The pipeline can be:

```text
Camera Image
      ↓
Local Face Detection
      ↓
Face Bounding Box
      ↓
Correct Crop/Alignment
      ↓
MiVOLO
      ↓
Age + Gender
```

However, do not replace the official MiVOLO preprocessing pipeline with a random crop if MiVOLO requires a specific preprocessing method.

Follow the official implementation as closely as possible.

---

# 11. Project structure

Use a clean but simple structure similar to:

```text
MiVOLOAgeGenderPOC/
│
├── app/
│   ├── src/main/
│   │   ├── java/.../
│   │   │   ├── MainActivity.kt
│   │   │   ├── CameraActivity.kt
│   │   │   ├── MiVOLOProcessor.kt
│   │   │   ├── ImagePreprocessor.kt
│   │   │   └── ModelManager.kt
│   │   │
│   │   ├── assets/
│   │   │   └── model files
│   │   │
│   │   └── res/
│   │
├── scripts/
│   ├── download_models.py
│   ├── export_model.py
│   └── test_mivolo.py
│
├── README.md
└── build.gradle
```

You may adjust the structure if needed, but keep it simple.

---

# 12. README is mandatory

Create a detailed README explaining exactly how to run the POC.

Include:

## Prerequisites

```text
Android Studio version
Android SDK requirements
JDK version
Python version if model export is required
```

## Step 1

Clone/open project.

## Step 2

Automatically download the required MiVOLO model/checkpoint.

## Step 3

Export/convert the model if required.

## Step 4

Place/package the final model into Android assets.

## Step 5

Open Android Studio.

## Step 6

Build and run.

Also explain how to test the app with the internet completely disabled after setup.

---

# 13. Add debug logging

During the POC stage, add useful logs.

For example:

```text
MiVOLO model loaded

Input image size: XXXX

Face detected

Face crop: XXXX

Model input shape: [1, 3, H, W]

Inference started

Inference completed

Raw model output: ...

Decoded age: ...

Decoded gender: ...
```

Use Android Logcat.

This is important because I need to debug the model integration.

---

# 14. Accuracy and preprocessing are important

Do not oversimplify the model pipeline.

Accuracy is more important than making the code look short.

Make sure the implementation matches the official MiVOLO preprocessing.

Specifically verify:

* Input resolution
* RGB or BGR
* Tensor format: NCHW/NHWC
* Normalization mean
* Normalization standard deviation
* Face crop strategy
* Padding strategy
* Output tensor structure
* Age decoding
* Gender decoding

Do not guess any of these.

Inspect the official MiVOLO code and reproduce the actual inference pipeline.

---

# 15. Do not add these features yet

Do NOT add:

* Continuous video inference
* Multiple person tracking
* Face recognition
* Database
* Login
* API/backend
* Cloud processing
* Analytics dashboard
* History
* User accounts
* Complex UI
* Multiple age models

This is a very simple POC.

---

# Final Deliverable

I expect a complete Android project that can be opened in Android Studio.

The final POC should work like this:

```text
┌──────────────────────────┐
│                          │
│   MiVOLO Age & Gender    │
│                          │
│       [ MiVOLO ]         │
│                          │
└──────────────────────────┘

            ↓

┌──────────────────────────┐
│                          │
│      CAMERA PREVIEW      │
│                          │
│       [ Capture ]        │
│                          │
└──────────────────────────┘

            ↓

┌──────────────────────────┐
│                          │
│     Captured Image       │
│                          │
│     Analyzing...         │
│                          │
│     Age: 27              │
│     Gender: Male         │
│                          │
│   [ Capture Again ]      │
│                          │
└──────────────────────────┘
```

---

# Very Important Implementation Rules

1. First inspect the current official MiVOLO repository and understand how inference actually works.
2. Do not invent model URLs or filenames.
3. Do not use any cloud API.
4. The final inference must run locally on the Android device.
5. Automatically download all required open-source model files where possible.
6. Validate the model in Python before Android integration.
7. Match Python and Android preprocessing exactly.
8. Prefer correctness over unnecessary architecture.
9. If MiVOLO cannot be directly deployed to Android in its original form, implement the necessary export/conversion pipeline and include scripts.
10. Do not stop at pseudocode. Generate the actual working project files and code.
11. If a particular export path fails, debug it and choose another compatible runtime rather than leaving placeholder code.
12. Clearly tell me which exact model file is being used and where it comes from.

Start by inspecting the MiVOLO repository and then build the POC step by step.
