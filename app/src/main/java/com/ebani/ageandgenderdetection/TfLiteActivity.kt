package com.ebani.ageandgenderdetection

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.ebani.ageandgenderdetection.ui.theme.AgeandgenderdetectionTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Screen 3: the TFLite path, live.
 *
 * Same shape as [CameraActivity] -- no shutter, every face in every frame gets a
 * labelled box -- but every number comes from the `.tflite` graphs in
 * `assets/tflite/` via [TfLiteProcessor]. The two screens share no models, no
 * metadata and no state; deleting this file and its manifest entry leaves the
 * MiVOLO flow untouched.
 */
class TfLiteActivity : ComponentActivity() {

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AgeandgenderdetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    TfLiteScreen(
                        contentPadding = innerPadding,
                        cameraExecutor = cameraExecutor,
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
        /** Kept for source compatibility; the live screen has no self-test path. */
        const val EXTRA_RUN_SELF_TEST = "run_self_test"
    }
}

@Composable
private fun TfLiteScreen(
    contentPadding: PaddingValues,
    cameraExecutor: ExecutorService,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Constructing the processor parses tflite_meta.json, so a malformed or
    // missing metadata file surfaces as a message rather than a crash.
    val processor = remember { runCatching { TfLiteProcessor(context.applicationContext) } }

    val failure = processor.exceptionOrNull()
    if (failure != null) {
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
                    (failure.message ?: failure::class.java.simpleName),
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
    var overlay by remember { mutableStateOf(OverlayFrame()) }
    var status by remember { mutableStateOf("Starting camera...") }
    var lensFacing by remember { mutableStateOf<Int?>(null) }
    var useFrontCamera by remember { mutableStateOf(true) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    if (!hasCameraPermission) {
        PermissionPrompt(contentPadding) { permissionLauncher.launch(Manifest.permission.CAMERA) }
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LiveCameraPreview(
                modifier = Modifier.fillMaxSize(),
                lifecycleOwner = lifecycleOwner,
                useFrontCamera = useFrontCamera,
                executor = cameraExecutor,
                analysisResolution = ANALYSIS_RESOLUTION,
                onLensFacing = { lensFacing = it },
                onFrame = { bitmap ->
                    // The frame's own dimensions are the overlay's coordinate
                    // space, so they are read before the bitmap is released.
                    val frameWidth = bitmap.width
                    val frameHeight = bitmap.height

                    val started = System.nanoTime()
                    val faces = try {
                        engine.analyzeAll(bitmap, variantId)
                    } finally {
                        bitmap.recycle()
                    }
                    val millis = (System.nanoTime() - started) / 1_000_000

                    overlay = OverlayFrame(
                        faces = faces.map {
                            DetectedFace(
                                x1 = it.box.x1,
                                y1 = it.box.y1,
                                x2 = it.box.x2,
                                y2 = it.box.y2,
                                label = "${it.ageRounded} · ${it.genderDisplay}",
                            )
                        },
                        sourceWidth = frameWidth,
                        sourceHeight = frameHeight,
                    )
                    status =
                        if (faces.isEmpty()) "No face in view  ·  $millis ms"
                        else "${faces.size} face(s)  ·  $millis ms"
                },
            )

            FaceOverlay(
                modifier = Modifier.fillMaxSize(),
                frame = overlay,
                mirrored = lensFacing == CameraSelector.LENS_FACING_FRONT,
                boxColor = MaterialTheme.colorScheme.tertiary,
            )
        }

        Spacer(Modifier.height(12.dp))

        Text(
            text = status,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(12.dp))

        // The variants come from tflite_meta.json, so adding a third build of
        // the models adds a third chip here with no code change.
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

        OutlinedButton(
            modifier = Modifier.fillMaxWidth(),
            onClick = { useFrontCamera = !useFrontCamera },
        ) {
            Text(if (useFrontCamera) "Switch to back camera" else "Switch to front camera")
        }
    }
}

/**
 * 480x640 is what the detector actually consumes: YuNet letterboxes onto a
 * 640x640 canvas, so a larger frame costs a downscale and buys no recall, while
 * every extra pixel is paid for on the single analysis thread.
 */
private val ANALYSIS_RESOLUTION = Size(480, 640)
