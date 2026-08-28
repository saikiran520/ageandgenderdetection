package com.ebani.ageandgenderdetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.ebani.ageandgenderdetection.ui.theme.AgeandgenderdetectionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import androidx.lifecycle.lifecycleScope

/**
 * Screen 2: camera preview, one Capture button, then the result.
 *
 * All inference happens on Dispatchers.Default; the UI thread only ever renders
 * the state.
 */
class CameraActivity : ComponentActivity() {

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val runSelfTest = intent.getBooleanExtra(EXTRA_RUN_SELF_TEST, false)

        setContent {
            AgeandgenderdetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CameraScreen(
                        contentPadding = innerPadding,
                        cameraExecutor = cameraExecutor,
                        runSelfTestOnStart = runSelfTest,
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        const val EXTRA_RUN_SELF_TEST = "run_self_test"
    }
}

/** What the screen is currently showing. */
private sealed interface ScreenState {
    data object Previewing : ScreenState
    data object Analyzing : ScreenState
    data class Done(val image: Bitmap?, val result: MiVOLOProcessor.Result) : ScreenState
}

@Composable
private fun CameraScreen(
    contentPadding: PaddingValues,
    cameraExecutor: java.util.concurrent.ExecutorService,
    runSelfTestOnStart: Boolean,
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val lifecycleOwner = LocalLifecycleOwner.current

    var state by remember { mutableStateOf<ScreenState>(ScreenState.Previewing) }
    var useFrontCamera by remember { mutableStateOf(true) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val processor = remember { MiVOLOProcessor(context.applicationContext) }

    // Cap the capture at roughly 1080p. A modern sensor will happily hand back a
    // 12 MP JPEG, which decodes to a ~48 MB bitmap and then gets rotated into a
    // second one -- enough to OOM a mid-range device for no benefit, since the
    // detector works on a 640x640 canvas and MiVOLO on a 224x224 crop.
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(1080, 1440),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                        )
                    )
                    .build()
            )
            .build()
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission && !runSelfTestOnStart) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // The self-test path never touches the camera: it decodes assets/test.jpg
    // and runs the identical pipeline, so Python and Android can be diffed.
    LaunchedEffect(runSelfTestOnStart) {
        if (!runSelfTestOnStart) return@LaunchedEffect
        state = ScreenState.Analyzing
        val outcome = withContext(Dispatchers.Default) { processor.runSelfTest() }
        state = ScreenState.Done(null, outcome)
    }

    DisposableEffect(Unit) {
        onDispose { }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        when (val current = state) {
            ScreenState.Previewing, ScreenState.Analyzing -> {
                if (runSelfTestOnStart) {
                    Text("Self-test", style = MaterialTheme.typography.titleMedium)
                } else if (!hasCameraPermission) {
                    Text(
                        "Camera permission is required to capture a photo.",
                        textAlign = TextAlign.Center,
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                        Text("Grant camera permission")
                    }
                } else {
                    CameraPreview(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(3f / 4f),
                        lifecycleOwner = lifecycleOwner,
                        imageCapture = imageCapture,
                        useFrontCamera = useFrontCamera,
                    )
                    Spacer(Modifier.height(16.dp))

                    if (current is ScreenState.Analyzing) {
                        AnalyzingRow()
                    } else {
                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            onClick = {
                                state = ScreenState.Analyzing
                                capture(
                                    imageCapture = imageCapture,
                                    executor = cameraExecutor,
                                    mirrored = useFrontCamera,
                                ) { bitmap ->
                                    activity.lifecycleScope.launch {
                                        if (bitmap == null) {
                                            state = ScreenState.Done(
                                                null,
                                                MiVOLOProcessor.Result.Failure("Capture failed", null),
                                            )
                                            return@launch
                                        }
                                        val outcome = withContext(Dispatchers.Default) {
                                            processor.analyze(bitmap)
                                        }
                                        state = ScreenState.Done(bitmap, outcome)
                                    }
                                }
                            },
                        ) {
                            Text("Capture")
                        }
                        Spacer(Modifier.height(8.dp))
                        OutlinedButton(
                            modifier = Modifier.fillMaxWidth(),
                            onClick = { useFrontCamera = !useFrontCamera },
                        ) {
                            Text(if (useFrontCamera) "Switch to back camera" else "Switch to front camera")
                        }
                    }
                }
            }

            is ScreenState.Done -> {
                current.image?.let { bitmap ->
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Captured image",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(3f / 4f),
                    )
                    Spacer(Modifier.height(16.dp))
                }

                ResultBlock(current.result)

                Spacer(Modifier.height(24.dp))
                Button(
                    modifier = Modifier.fillMaxWidth(),
                    onClick = {
                        current.image?.recycle()
                        state = ScreenState.Previewing
                    },
                ) {
                    Text("Capture Again")
                }
            }
        }
    }
}

@Composable
private fun AnalyzingRow() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(12.dp))
        Text("Analyzing...", style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
private fun ResultBlock(result: MiVOLOProcessor.Result) {
    when (result) {
        is MiVOLOProcessor.Result.Success -> {
            Text(
                "Age: ${result.ageRounded}",
                style = MaterialTheme.typography.headlineMedium,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Gender: ${result.genderDisplay}",
                style = MaterialTheme.typography.headlineMedium,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                "detect ${result.detectMillis} ms  ·  inference ${result.inferenceMillis} ms  " +
                    "·  total ${result.totalMillis} ms",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        is MiVOLOProcessor.Result.NoFace -> {
            Text(
                "No face detected.\nPlease capture a clear face.",
                style = MaterialTheme.typography.titleLarge,
                textAlign = TextAlign.Center,
            )
        }

        is MiVOLOProcessor.Result.Failure -> {
            Text(
                "Analysis failed.\n${result.message}",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun CameraPreview(
    modifier: Modifier,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    imageCapture: ImageCapture,
    useFrontCamera: Boolean,
) {
    val context = LocalContext.current
    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
            },
            update = { view ->
                val providerFuture = ProcessCameraProvider.getInstance(context)
                providerFuture.addListener({
                    runCatching {
                        val provider = providerFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.surfaceProvider = view.surfaceProvider
                        }
                        val selector = if (useFrontCamera) {
                            CameraSelector.DEFAULT_FRONT_CAMERA
                        } else {
                            CameraSelector.DEFAULT_BACK_CAMERA
                        }
                        provider.unbindAll()
                        provider.bindToLifecycle(lifecycleOwner, selector, preview, imageCapture)
                    }.onFailure {
                        Log.e(ModelManager.TAG, "Camera bind failed", it)
                    }
                }, ContextCompat.getMainExecutor(context))
            },
        )
    }
}

/**
 * Take one still and hand back an upright bitmap.
 *
 * ImageCapture yields JPEG bytes plus a rotation in metadata; the rotation has
 * to be applied before detection, because a sideways face is a much harder
 * detection problem than an upright one.
 */
private fun capture(
    imageCapture: ImageCapture,
    executor: java.util.concurrent.ExecutorService,
    mirrored: Boolean,
    onResult: (Bitmap?) -> Unit,
) {
    imageCapture.takePicture(
        executor,
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bitmap = runCatching { image.toUprightBitmap(mirrored) }
                    .onFailure { Log.e(ModelManager.TAG, "Failed to decode capture", it) }
                    .getOrNull()
                image.close()
                onResult(bitmap)
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(ModelManager.TAG, "Capture failed", exception)
                onResult(null)
            }
        },
    )
}

private fun ImageProxy.toUprightBitmap(mirrored: Boolean): Bitmap {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }
    val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        ?: throw IllegalStateException("BitmapFactory returned null for ${bytes.size} JPEG bytes")

    val rotation = imageInfo.rotationDegrees
    if (rotation == 0 && !mirrored) return decoded

    val matrix = Matrix().apply {
        if (rotation != 0) postRotate(rotation.toFloat())
        // Un-mirror the front camera so the saved image matches what the user saw.
        // Age and gender are unaffected by the flip; this only keeps the preview
        // and the captured image consistent.
        if (mirrored) postScale(-1f, 1f)
    }
    val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
    if (rotated !== decoded) decoded.recycle()
    return rotated
}

@Suppress("unused")
private fun Bitmap.toJpegBytes(): ByteArray =
    ByteArrayOutputStream().also { compress(Bitmap.CompressFormat.JPEG, 95, it) }.toByteArray()
