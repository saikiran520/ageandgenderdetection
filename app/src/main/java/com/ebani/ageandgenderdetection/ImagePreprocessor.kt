package com.ebani.ageandgenderdetection

import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Turns a face crop into the exact tensor MiVOLO expects.
 *
 * This is a deliberate line-for-line port of the official pipeline, not an
 * approximation of it:
 *
 *   mivolo/data/misc.py :: class_letterbox
 *       resize preserving aspect ratio, then pad to a square with black,
 *       splitting the padding between the two sides.
 *   mivolo/data/misc.py :: prepare_classification_images
 *       BGR -> RGB, divide by 255, subtract ImageNet mean, divide by ImageNet
 *       std, transpose HWC -> CHW, float32.
 *
 * Two details are worth calling out, because getting either wrong shifts the
 * predicted age by years rather than decimals:
 *
 *  1. The bilinear resize below reimplements OpenCV's INTER_LINEAR sampling grid
 *     (`src = (dst + 0.5) * scale - 0.5`) rather than calling
 *     Bitmap.createScaledBitmap. Android's scaler uses a different sample
 *     placement, which would make the Android tensor drift from the Python one
 *     even though both are nominally "bilinear".
 *  2. The padding is asymmetric in exactly the way OpenCV is: `top` uses
 *     round(dh - 0.1) and `bottom` uses round(dh + 0.1), so an odd number of
 *     padding rows lands one extra row at the bottom.
 */
object ImagePreprocessor {

    private const val TAG = ModelManager.TAG

    /** Result of preprocessing, plus a fingerprint used to diff against Python. */
    class Tensor(
        val buffer: FloatBuffer,
        val shape: LongArray,
        val sum: Double,
        val mean: Double,
        val min: Float,
        val max: Float,
        val first8: FloatArray,
    ) {
        fun describe(): String =
            "shape=[${shape.joinToString()}] sum=${"%.4f".format(sum)} mean=${"%.6f".format(mean)} " +
                "min=${"%.6f".format(min)} max=${"%.6f".format(max)} " +
                "first8=[${first8.joinToString { "%.6f".format(it) }}]"
    }

    /**
     * @param crop the face crop, ARGB_8888
     * @param meta drives target size, channel count and normalisation
     */
    fun prepare(crop: Bitmap, meta: ModelManager.ModelMeta): Tensor {
        val size = meta.inputSize
        val boxed = letterbox(crop, size)
        Log.d(TAG, "Letterboxed crop ${crop.width}x${crop.height} -> ${size}x$size")

        val pixels = IntArray(size * size)
        boxed.getPixels(pixels, 0, size, 0, 0, size, size)
        if (boxed !== crop) boxed.recycle()

        val plane = size * size
        val channels = meta.inChannels
        val values = FloatArray(channels * plane)

        val scale = meta.pixelScale
        // Bitmap pixels are already RGB, so we index the RGB channels directly.
        // MiVOLO reaches the same place via cv2.cvtColor(BGR2RGB) on a BGR crop.
        for (i in 0 until plane) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / scale
            val g = ((pixel shr 8) and 0xFF) / scale
            val b = (pixel and 0xFF) / scale
            values[i] = (r - meta.mean[0]) / meta.std[0]
            values[plane + i] = (g - meta.mean[1]) / meta.std[1]
            values[2 * plane + i] = (b - meta.mean[2]) / meta.std[2]
        }

        if (channels == 6) {
            // A face+person checkpoint driven face-only. MiVOLO fills a missing
            // crop with a zero image put through the same normalisation
            // (prepare_classification_images, the `img is None` branch), which
            // is a constant -mean/std per channel, not zeros.
            for (c in 0 until 3) {
                val fill = (0f - meta.mean[c]) / meta.std[c]
                java.util.Arrays.fill(values, (3 + c) * plane, (4 + c) * plane, fill)
            }
        }

        var sum = 0.0
        var minimum = Float.MAX_VALUE
        var maximum = -Float.MAX_VALUE
        for (v in values) {
            sum += v
            if (v < minimum) minimum = v
            if (v > maximum) maximum = v
        }

        return Tensor(
            buffer = FloatBuffer.wrap(values),
            shape = longArrayOf(1, channels.toLong(), size.toLong(), size.toLong()),
            sum = sum,
            mean = sum / values.size,
            min = minimum,
            max = maximum,
            first8 = values.copyOfRange(0, min(8, values.size)),
        )
    }

    /**
     * Port of mivolo/data/misc.py :: class_letterbox with scaleup = true.
     *
     * ```python
     * r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
     * new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
     * dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
     * dw /= 2; dh /= 2
     * im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
     * top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
     * left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
     * ```
     */
    fun letterbox(source: Bitmap, target: Int): Bitmap {
        if (source.width == target && source.height == target) return source

        val r = min(target.toFloat() / source.height, target.toFloat() / source.width)
        val unpadW = (source.width * r).roundToInt()
        val unpadH = (source.height * r).roundToInt()

        val dw = (target - unpadW) / 2f
        val dh = (target - unpadH) / 2f
        val left = pythonRound(dw - 0.1f)
        val top = pythonRound(dh - 0.1f)

        val resized = resizeBilinear(source, unpadW, unpadH)

        // BORDER_CONSTANT with value (0,0,0): opaque black, matching MiVOLO.
        val out = Bitmap.createBitmap(target, target, Bitmap.Config.ARGB_8888)
        out.eraseColor(0xFF000000.toInt())

        val row = IntArray(unpadW)
        for (y in 0 until unpadH) {
            resized.getPixels(row, 0, unpadW, 0, y, unpadW, 1)
            out.setPixels(row, 0, unpadW, left, top + y, unpadW, 1)
        }
        if (resized !== source) resized.recycle()
        return out
    }

    /**
     * cv2.resize(..., interpolation=cv2.INTER_LINEAR) reimplemented.
     *
     * OpenCV maps destination pixel centres to source pixel centres with
     * `src = (dst + 0.5) * scale - 0.5` and clamps at the borders. Android's own
     * bilinear scaler does not use this grid, so a straight
     * Bitmap.createScaledBitmap would quietly diverge from the Python reference.
     */
    fun resizeBilinear(source: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        if (source.width == targetWidth && source.height == targetHeight) return source

        val srcW = source.width
        val srcH = source.height
        val srcPixels = IntArray(srcW * srcH)
        source.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

        val scaleX = srcW.toDouble() / targetWidth
        val scaleY = srcH.toDouble() / targetHeight
        val out = IntArray(targetWidth * targetHeight)

        // Precompute the horizontal taps: they are identical for every row.
        val x0s = IntArray(targetWidth)
        val x1s = IntArray(targetWidth)
        val xFracs = DoubleArray(targetWidth)
        for (x in 0 until targetWidth) {
            val fx = (x + 0.5) * scaleX - 0.5
            val ix = floor(fx).toInt()
            val frac = fx - ix
            x0s[x] = ix.coerceIn(0, srcW - 1)
            x1s[x] = (ix + 1).coerceIn(0, srcW - 1)
            xFracs[x] = if (ix < 0) 0.0 else if (ix >= srcW - 1) 0.0 else frac
        }

        for (y in 0 until targetHeight) {
            val fy = (y + 0.5) * scaleY - 0.5
            val iy = floor(fy).toInt()
            val yFracRaw = fy - iy
            val y0 = iy.coerceIn(0, srcH - 1)
            val y1 = (iy + 1).coerceIn(0, srcH - 1)
            val yFrac = if (iy < 0) 0.0 else if (iy >= srcH - 1) 0.0 else yFracRaw

            val rowBase0 = y0 * srcW
            val rowBase1 = y1 * srcW
            val outBase = y * targetWidth

            for (x in 0 until targetWidth) {
                val p00 = srcPixels[rowBase0 + x0s[x]]
                val p01 = srcPixels[rowBase0 + x1s[x]]
                val p10 = srcPixels[rowBase1 + x0s[x]]
                val p11 = srcPixels[rowBase1 + x1s[x]]
                val xf = xFracs[x]

                var packed = 0xFF shl 24
                var shift = 16
                while (shift >= 0) {
                    val c00 = (p00 shr shift) and 0xFF
                    val c01 = (p01 shr shift) and 0xFF
                    val c10 = (p10 shr shift) and 0xFF
                    val c11 = (p11 shr shift) and 0xFF
                    val top = c00 + (c01 - c00) * xf
                    val bottom = c10 + (c11 - c10) * xf
                    val value = top + (bottom - top) * yFrac
                    // cv2 rounds half away from zero when writing back to uint8.
                    val rounded = floor(value + 0.5).toInt().coerceIn(0, 255)
                    packed = packed or (rounded shl shift)
                    shift -= 8
                }
                out[outBase + x] = packed
            }
        }

        return Bitmap.createBitmap(out, targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
    }

    /**
     * Python's round() uses banker's rounding, and OpenCV's letterbox relies on
     * it via int(round(dh - 0.1)). The -0.1/+0.1 offsets mean ties never occur
     * in practice, but matching the semantics keeps the port honest.
     */
    private fun pythonRound(value: Float): Int {
        val floored = floor(value.toDouble())
        val diff = value - floored
        return when {
            diff > 0.5 -> (floored + 1).toInt()
            diff < 0.5 -> floored.toInt()
            else -> { // exact tie: round to even
                val f = floored.toInt()
                if (f % 2 == 0) f else f + 1
            }
        }
    }

    /** Crop with clamping, mirroring PersonAndFaceResult.get_bbox_by_ind + crop_object. */
    fun cropClamped(source: Bitmap, x1: Int, y1: Int, x2: Int, y2: Int): Bitmap? {
        val left = max(0, min(x1, source.width - 1))
        val top = max(0, min(y1, source.height - 1))
        val right = max(0, min(x2, source.width - 1))
        val bottom = max(0, min(y2, source.height - 1))
        if (right <= left || bottom <= top) return null
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }
}
