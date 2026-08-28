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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.ebani.ageandgenderdetection.ui.theme.AgeandgenderdetectionTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Screen 2: the MiVOLO path, live.
 *
 * There is no shutter. Frames stream off the camera continuously, every face in
 * each frame is measured, and the answers are drawn as labelled boxes over the
 * preview. [TfLiteActivity] does the same thing with different models; the two
 * share no state, no metadata and no results.
 */
class CameraActivity : ComponentActivity() {

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AgeandgenderdetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CameraScreen(
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
private fun CameraScreen(
    contentPadding: PaddingValues,
    cameraExecutor: ExecutorService,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val processor = remember { MiVOLOProcessor(context.applicationContext) }

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
                        processor.analyzeAll(bitmap)
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
                boxColor = MaterialTheme.colorScheme.primary,
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

        OutlinedButton(
            modifier = Modifier.fillMaxWidth(),
            onClick = { useFrontCamera = !useFrontCamera },
        ) {
            Text(if (useFrontCamera) "Switch to back camera" else "Switch to front camera")
        }
    }
}

@Composable
internal fun PermissionPrompt(contentPadding: PaddingValues, onRequest: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Camera permission is required for live detection.", textAlign = TextAlign.Center)
        Spacer(Modifier.height(16.dp))
        Button(onClick = onRequest) { Text("Grant camera permission") }
    }
}

/**
 * 640x480 is what the detector actually consumes: YuNet letterboxes onto a
 * 640x640 canvas, so a larger frame costs a downscale and buys no recall, while
 * every extra pixel is paid for on the single analysis thread.
 */
private val ANALYSIS_RESOLUTION = Size(480, 640)
