package com.ebani.ageandgenderdetection

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as GeometrySize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import kotlin.math.max

/**
 * The CameraX plumbing both live screens need.
 *
 * [CameraActivity] (MiVOLO / ONNX) and [TfLiteActivity] (TFLite) run different
 * models and show different results, but "stream frames" and "draw boxes over
 * the preview" are the same problem in both, so they live here once rather than
 * being copied. No model, metadata or result is shared between them.
 */

/** One face to draw, in the coordinate space of the analysed frame. */
data class DetectedFace(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val label: String,
)

/** A frame's worth of results, plus the size of the frame they were measured in. */
data class OverlayFrame(
    val faces: List<DetectedFace> = emptyList(),
    val sourceWidth: Int = 0,
    val sourceHeight: Int = 0,
)

/**
 * Live preview with a frame analyser attached.
 *
 * Frames arrive on [executor] as upright bitmaps -- already rotated out of the
 * sensor orientation, because a sideways face is a much harder detection problem
 * than an upright one. The analyser owns the bitmap and must recycle it.
 *
 * Backpressure is KEEP_ONLY_LATEST, so a slow model cannot build a queue: frames
 * arriving while [onFrame] is still working are dropped, and the next analysis
 * starts from whatever the camera is showing by then. That is what keeps the
 * preview smooth while a 25 M parameter model takes its time behind it.
 *
 * @param onLensFacing reports the lens actually bound, so the caller knows
 *   whether PreviewView is mirroring and can flip its overlay to match.
 */
@Composable
internal fun LiveCameraPreview(
    modifier: Modifier,
    lifecycleOwner: LifecycleOwner,
    useFrontCamera: Boolean,
    executor: ExecutorService,
    analysisResolution: Size,
    onLensFacing: (Int?) -> Unit,
    onFrame: (Bitmap) -> Unit,
) {
    val context = LocalContext.current

    // The analyser callback is rebuilt on every recomposition -- it closes over
    // screen state such as the selected model variant. Holding it behind
    // rememberUpdatedState lets the binding below stay put while the callback it
    // invokes is always the newest one.
    val currentOnFrame by rememberUpdatedState(onFrame)
    val currentOnLensFacing by rememberUpdatedState(onLensFacing)

    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }

    // Keyed on the lens only. Binding is expensive and tears down the capture
    // session, so it must not happen when the overlay redraws -- which is every
    // analysed frame.
    DisposableEffect(useFrontCamera, lifecycleOwner) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            runCatching {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                // A modest analysis resolution keeps detection cheap: YuNet
                // letterboxes onto a 640x640 canvas anyway, so a 12 MP frame
                // would buy a large downscale and no extra recall.
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(
                                AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
                            )
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    analysisResolution,
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    )
                    .build()
                    .also { it.setAnalyzer(executor) { proxy -> proxy.dispatch { currentOnFrame(it) } } }

                provider.unbindAll()
                val camera = provider.bindToLifecycle(
                    lifecycleOwner,
                    selectCamera(provider, useFrontCamera),
                    preview,
                    analysis,
                )
                currentOnLensFacing(camera.cameraInfo.lensFacing)
            }.onFailure {
                Log.e(ModelManager.TAG, "Camera bind failed", it)
                currentOnLensFacing(null)
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            runCatching { providerFuture.get().unbindAll() }
        }
    }

    Box(modifier = modifier) {
        AndroidView(modifier = Modifier.fillMaxSize(), factory = { previewView })
    }
}

/** Hand the analyser an upright bitmap, and always close the proxy. */
private fun ImageProxy.dispatch(onFrame: (Bitmap) -> Unit) {
    try {
        val rotation = imageInfo.rotationDegrees
        val raw = toBitmap()
        val upright = if (rotation == 0) {
            raw
        } else {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
                .also { if (it !== raw) raw.recycle() }
        }
        onFrame(upright)
    } catch (t: Throwable) {
        Log.e(ModelManager.TAG, "Frame analysis failed", t)
    } finally {
        close()
    }
}

/**
 * Draw one box and label per detected face, on top of the preview.
 *
 * PreviewView is FILL_CENTER, so the frame is scaled by the *larger* of the two
 * ratios and the overflow is cropped evenly on both sides. The overlay has to
 * reproduce that exactly, or the boxes drift off the faces as soon as the
 * preview's aspect ratio differs from the analyser's.
 *
 * @param mirrored true when PreviewView is mirroring the feed, which it does for
 *   a front-facing lens only. Analysis frames are never mirrored, so the boxes
 *   have to be flipped to land on the face the user can actually see.
 */
@Composable
internal fun FaceOverlay(
    modifier: Modifier,
    frame: OverlayFrame,
    mirrored: Boolean,
    boxColor: Color,
) {
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier) {
        if (frame.sourceWidth <= 0 || frame.sourceHeight <= 0) return@Canvas

        val scale = max(
            size.width / frame.sourceWidth,
            size.height / frame.sourceHeight,
        )
        val offsetX = (size.width - frame.sourceWidth * scale) / 2f
        val offsetY = (size.height - frame.sourceHeight * scale) / 2f

        for (face in frame.faces) {
            val rawLeft = face.x1 * scale + offsetX
            val rawRight = face.x2 * scale + offsetX
            val top = face.y1 * scale + offsetY
            val bottom = face.y2 * scale + offsetY

            // Flipping swaps which edge ends up on the left.
            val left = if (mirrored) size.width - rawRight else rawLeft
            val right = if (mirrored) size.width - rawLeft else rawRight

            drawRect(
                color = boxColor,
                topLeft = Offset(left, top),
                size = GeometrySize(right - left, bottom - top),
                style = Stroke(width = 3.dp.toPx()),
            )

            val measured = textMeasurer.measure(
                text = face.label,
                style = TextStyle(fontSize = 14.sp, color = Color.White),
            )
            val labelWidth = measured.size.width.toFloat()
            val labelHeight = measured.size.height.toFloat()
            val padding = 6.dp.toPx()

            // Prefer the label above the box, but drop it inside when there is
            // no room, so a face near the top edge stays readable.
            val labelTop =
                if (top - labelHeight - padding >= 0f) top - labelHeight - padding else top

            drawRect(
                color = boxColor,
                topLeft = Offset(left, labelTop),
                size = GeometrySize(labelWidth + padding * 2, labelHeight + padding / 2),
            )
            drawText(
                textLayoutResult = measured,
                topLeft = Offset(left + padding, labelTop),
            )
        }
    }
}

/**
 * The camera to bind: the one asked for, the other one, or failing both, any
 * camera at all.
 *
 * A phone has a front and a back camera and the first branch always wins. But
 * plenty of devices have neither as CameraX defines them -- a tablet or set-top
 * box with a single USB camera reports lens-facing EXTERNAL, which matches
 * DEFAULT_FRONT_CAMERA and DEFAULT_BACK_CAMERA equally badly, and binding one of
 * those throws "No available camera can be found". An unfiltered CameraSelector
 * matches anything, so the POC still gets a preview there.
 */
private fun selectCamera(
    provider: ProcessCameraProvider,
    useFrontCamera: Boolean,
): CameraSelector {
    val preferred =
        if (useFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
        else CameraSelector.DEFAULT_BACK_CAMERA
    val fallback =
        if (useFrontCamera) CameraSelector.DEFAULT_BACK_CAMERA
        else CameraSelector.DEFAULT_FRONT_CAMERA

    return when {
        runCatching { provider.hasCamera(preferred) }.getOrDefault(false) -> preferred

        runCatching { provider.hasCamera(fallback) }.getOrDefault(false) -> {
            Log.w(ModelManager.TAG, "Requested camera is absent; using the other one")
            fallback
        }

        else -> {
            Log.w(
                ModelManager.TAG,
                "Neither front nor back camera present; binding the first available of " +
                    "${provider.availableCameraInfos.size}",
            )
            CameraSelector.Builder().build()
        }
    }
}
