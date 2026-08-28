package com.ebani.ageandgenderdetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.ebani.ageandgenderdetection.ui.theme.AgeandgenderdetectionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Screen 3: the TFLite path. Same shape as [CameraActivity] -- preview, one
 * Capture button, then Age and Gender -- but every number comes from the
 * `.tflite` graphs in `assets/tflite/` via [TfLiteProcessor].
 *
 * The two paths share no state and no models beyond the YuNet face detector.
 * Deleting this file and its manifest entry leaves the MiVOLO flow untouched.
 */
class TfLiteActivity : ComponentActivity() {

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val runSelfTest = intent.getBooleanExtra(EXTRA_RUN_SELF_TEST, false)

        setContent {
            AgeandgenderdetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    TfLiteScreen(
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
        /**
         * Not part of the POC flow. Runs assets/test.jpg through the identical
         * pipeline instead of opening the camera, so the graphs, preprocessing
         * and decoding can be checked from adb:
         *
         *     adb shell am start -n <pkg>/.TfLiteActivity --ez run_self_test true
         *
         * Mirrors CameraActivity.EXTRA_RUN_SELF_TEST.
         */
        const val EXTRA_RUN_SELF_TEST = "run_self_test"
    }
}

private sealed interface TfLiteScreenState {
    data object Previewing : TfLiteScreenState
    data object Analyzing : TfLiteScreenState
    data class Done(val image: Bitmap?, val result: TfLiteProcessor.Result) : TfLiteScreenState
    data class Broken(val message: String) : TfLiteScreenState
}

@Composable
private fun TfLiteScreen(
    contentPadding: PaddingValues,
    cameraExecutor: ExecutorService,
    runSelfTestOnStart: Boolean,
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val lifecycleOwner = LocalLifecycleOwner.current

    // Constructing the processor parses tflite_meta.json, so a malformed or
    // missing metadata file surfaces as a message rather than a crash.
    val processor = remember {
        runCatching { TfLiteProcessor(context.applicationContext) }
    }

    var state by remember {
        mutableStateOf<TfLiteScreenState>(
            processor.exceptionOrNull()?.let {
                TfLiteScreenState.Broken(it.message ?: it::class.java.simpleName)
            } ?: TfLiteScreenState.Previewing
        )
    }

    if (state is TfLiteScreenState.Broken) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(contentPadding)
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "TFLite models could not be loaded.\n" +
                    (state as TfLiteScreenState.Broken).message,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
        }
        return
    }

    val engine = processor.getOrThrow()
    val variants = remember { engine.variants() }
    var variantId by remember { mutableStateOf(engine.defaultVariantId()) }
    var useFrontCamera by remember { mutableStateOf(true) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    // Same 1080p cap as the MiVOLO screen: a full-resolution sensor JPEG decodes
    // to tens of megabytes for no benefit, since the heads work on small crops.
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
    // and runs the identical pipeline.
    LaunchedEffect(runSelfTestOnStart) {
        if (!runSelfTestOnStart) return@LaunchedEffect
        state = TfLiteScreenState.Analyzing
        val selected = variantId
        val outcome = withContext(Dispatchers.Default) { engine.runSelfTest(selected) }
        state = TfLiteScreenState.Done(null, outcome)
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
            TfLiteScreenState.Previewing, TfLiteScreenState.Analyzing -> {
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
                    CameraPreviewSurface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(3f / 4f),
                        lifecycleOwner = lifecycleOwner,
                        imageCapture = imageCapture,
                        useFrontCamera = useFrontCamera,
                    )
                    Spacer(Modifier.height(16.dp))

                    if (current is TfLiteScreenState.Analyzing) {
                        CircularProgressIndicator()
                        Spacer(Modifier.height(12.dp))
                        Text("Analyzing...", style = MaterialTheme.typography.titleMedium)
                    } else {
                        // The variants come from tflite_meta.json, so adding a
                        // third build of the models adds a third chip here.
                        if (variants.size > 1) {
                            Row(horizontalArrangement = Arrangement.Center) {
                                variants.forEachIndexed { index, variant ->
                                    if (index > 0) Spacer(Modifier.width(8.dp))
                                    FilterChip(
                                        selected = variant.id == variantId,
                                        onClick = { variantId = variant.id },
                                        label = { Text(variant.label) },
                                    )
                                }
                            }
                            Spacer(Modifier.height(12.dp))
                        }

                        Button(
                            modifier = Modifier.fillMaxWidth(),
                            onClick = {
                                state = TfLiteScreenState.Analyzing
                                val selected = variantId
                                captureStill(
                                    imageCapture = imageCapture,
                                    executor = cameraExecutor,
                                    mirrored = useFrontCamera,
                                ) { bitmap ->
                                    activity.lifecycleScope.launch {
                                        if (bitmap == null) {
                                            state = TfLiteScreenState.Done(
                                                null,
                                                TfLiteProcessor.Result.Failure("Capture failed", null),
                                            )
                                            return@launch
                                        }
                                        val outcome = withContext(Dispatchers.Default) {
                                            engine.analyze(bitmap, selected)
                                        }
                                        state = TfLiteScreenState.Done(bitmap, outcome)
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
                            Text(
                                if (useFrontCamera) "Switch to back camera"
                                else "Switch to front camera"
                            )
                        }
                    }
                }
            }

            is TfLiteScreenState.Done -> {
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

                TfLiteResultBlock(current.result)

                Spacer(Modifier.height(24.dp))
                Button(
                    modifier = Modifier.fillMaxWidth(),
                    onClick = {
                        current.image?.recycle()
                        state = TfLiteScreenState.Previewing
                    },
                ) {
                    Text("Capture Again")
                }
            }

            is TfLiteScreenState.Broken -> Unit // handled above
        }
    }
}

@Composable
private fun TfLiteResultBlock(result: TfLiteProcessor.Result) {
    when (result) {
        is TfLiteProcessor.Result.Success -> {
            Text("Age: ${result.ageRounded}", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(8.dp))
            Text("Gender: ${result.genderDisplay}", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(12.dp))
            Text(
                "${result.variantLabel}  ·  detect ${result.detectMillis} ms  ·  " +
                    "inference ${result.inferenceMillis} ms  ·  total ${result.totalMillis} ms",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }

        is TfLiteProcessor.Result.NoFace -> {
            Text(
                "No face detected.\nPlease capture a clear face.",
                style = MaterialTheme.typography.titleLarge,
                textAlign = TextAlign.Center,
            )
        }

        is TfLiteProcessor.Result.Failure -> {
            Text(
                "Analysis failed.\n${result.message}",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
        }
    }
}
