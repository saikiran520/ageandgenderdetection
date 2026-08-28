package com.ebani.ageandgenderdetection

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService

/**
 * The CameraX plumbing both capture screens need.
 *
 * [CameraActivity] (MiVOLO / ONNX) and [TfLiteActivity] (TFLite) are separate
 * flows with separate models and separate results, but "show a preview" and
 * "take one upright still" are the same problem in both, so they live here once
 * rather than being copied.
 */

@Composable
internal fun CameraPreviewSurface(
    modifier: Modifier,
    lifecycleOwner: LifecycleOwner,
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
internal fun captureStill(
    imageCapture: ImageCapture,
    executor: ExecutorService,
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
