package com.ebani.ageandgenderdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The TFLite pipeline, end to end:
 *
 *     captured bitmap
 *        -> YuNet face detection             (FaceDetector, shared)
 *        -> pick the subject, add margin      (here)
 *        -> resize + normalise per head       (here)
 *        -> age .tflite and gender .tflite    (TfLiteModelManager)
 *        -> decode                            (here)
 *
 * This is entirely independent of [MiVOLOProcessor]: different graphs, different
 * runtime, different metadata file, its own screen. The one thing it reuses is
 * [FaceDetector], because face localisation is a separate problem from age and
 * gender estimation and the YuNet graph is already bundled and already offline.
 *
 * Nothing about the graphs is hardcoded here. The crop is resized to whatever
 * `spec.inputWidth`/`spec.inputHeight` the flatbuffer declares, the input buffer
 * is built for whatever `spec.inputType` it declares (dequantising on the way in
 * when the graph wants uint8/int8), and the answer is produced by whichever
 * [TfLiteModelManager.Decode] `tflite_meta.json` names for that head.
 */
class TfLiteProcessor(private val context: Context) {

    private val meta = TfLiteModelManager.meta(context)
    private val detector = FaceDetector(context)

    /** One head's answer plus how long it took. */
    data class Head(
        val label: String,
        val value: Float,
        val confidence: Float?,
        val raw: FloatArray,
        val millis: Long,
    )

    sealed interface Result {
        data class Success(
            val age: Float,
            val gender: String,
            val genderScore: Float,
            val variantLabel: String,
            val faceBox: FaceDetector.Box,
            val faceCount: Int,
            val detectMillis: Long,
            val inferenceMillis: Long,
            val totalMillis: Long,
        ) : Result {
            val ageRounded: Int get() = age.roundToInt()
            val genderDisplay: String get() = gender.replaceFirstChar { it.uppercase() }
        }

        data class NoFace(val reason: String) : Result

        data class Failure(val message: String, val cause: Throwable?) : Result
    }

    /** The variants `tflite_meta.json` offers, for the screen's selector. */
    fun variants(): List<TfLiteModelManager.Variant> = meta.variants

    fun defaultVariantId(): String = meta.defaultVariant

    /** One face's answer, for the live multi-face path. */
    data class Face(
        val box: FaceDetector.Box,
        val age: Float,
        val gender: String,
        val genderScore: Float,
    ) {
        val ageRounded: Int get() = age.roundToInt()
        val genderDisplay: String get() = gender.replaceFirstChar { it.uppercase() }
    }

    /**
     * Every face in the frame, each with its own age and gender.
     *
     * The detector runs once and both heads run once per face. A face that fails
     * to crop or infer is skipped rather than failing the frame, so one subject
     * leaving the shot does not blank the whole overlay.
     */
    fun analyzeAll(bitmap: Bitmap, variantId: String): List<Face> {
        val faces = detector.detect(bitmap)
        if (faces.isEmpty()) return emptyList()

        return faces.mapNotNull { raw ->
            val expanded = expand(raw)
            val crop = ImagePreprocessor.cropClamped(
                bitmap,
                expanded.x1.toInt(),
                expanded.y1.toInt(),
                expanded.x2.toInt(),
                expanded.y2.toInt(),
            ) ?: return@mapNotNull null

            try {
                val age = run(KEY_AGE, variantId, crop)
                val gender = run(KEY_GENDER, variantId, crop)
                Face(expanded, age.value, gender.label, gender.confidence ?: 0f)
            } catch (t: Throwable) {
                Log.e(TAG, "Per-face inference failed", t)
                null
            } finally {
                crop.recycle()
            }
        }
    }

    fun analyze(bitmap: Bitmap, variantId: String): Result {
        val startedAt = System.nanoTime()
        return try {
            Log.i(TAG, "===== tflite analysis start (variant $variantId) =====")
            Log.i(TAG, "Input image size: ${bitmap.width}x${bitmap.height}")

            val detectStart = System.nanoTime()
            val faces = detector.detect(bitmap)
            val detectMillis = (System.nanoTime() - detectStart) / 1_000_000

            val primary = detector.selectPrimary(faces)
                ?: return Result.NoFace("no face above the detector score threshold").also {
                    Log.w(TAG, "No face detected in ${bitmap.width}x${bitmap.height} image")
                }
            Log.i(TAG, "Face detected: $primary  (selected from ${faces.size})")

            val expanded = expand(primary)
            Log.i(TAG, "Face box after ${meta.faceCrop.marginX}/${meta.faceCrop.marginY} margin: $expanded")

            val crop = ImagePreprocessor.cropClamped(
                bitmap,
                expanded.x1.toInt(),
                expanded.y1.toInt(),
                expanded.x2.toInt(),
                expanded.y2.toInt(),
            ) ?: return Result.NoFace("face box fell outside the image").also {
                Log.w(TAG, "Expanded box did not intersect the image")
            }
            Log.i(TAG, "Face crop: ${crop.width}x${crop.height}")

            val inferenceStart = System.nanoTime()
            val heads = meta.models.keys.associateWith { key -> run(key, variantId, crop) }
            val inferenceMillis = (System.nanoTime() - inferenceStart) / 1_000_000
            crop.recycle()

            val age = heads[KEY_AGE]
                ?: return Result.Failure("${TfLiteModelManager.ASSET_META} declares no \"$KEY_AGE\" model", null)
            val gender = heads[KEY_GENDER]
                ?: return Result.Failure("${TfLiteModelManager.ASSET_META} declares no \"$KEY_GENDER\" model", null)

            val totalMillis = (System.nanoTime() - startedAt) / 1_000_000
            Log.i(TAG, "Decoded age    : ${age.value}")
            Log.i(TAG, "Decoded gender : ${gender.label} (${((gender.confidence ?: 0f) * 100).roundToInt()}%)")
            Log.i(TAG, "===== tflite analysis done in $totalMillis ms =====")

            Result.Success(
                age = age.value,
                gender = gender.label,
                genderScore = gender.confidence ?: 0f,
                variantLabel = meta.variant(variantId).label,
                faceBox = expanded,
                faceCount = faces.size,
                detectMillis = detectMillis,
                inferenceMillis = inferenceMillis,
                totalMillis = totalMillis,
            )
        } catch (t: Throwable) {
            Log.e(TAG, "TFLite analysis failed", t)
            Result.Failure(t.message ?: t::class.java.simpleName, t)
        }
    }

    /**
     * These heads were trained on UTKFace-style crops, which include hair and
     * chin, so the raw detector box is padded outwards by the fraction given in
     * `tflite_meta.json`. Set both margins to 0 there to feed the box as-is.
     */
    private fun expand(box: FaceDetector.Box): FaceDetector.Box {
        val dx = box.width * meta.faceCrop.marginX
        val dy = box.height * meta.faceCrop.marginY
        return FaceDetector.Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy, box.score)
    }

    /** Resize, normalise, run and decode one head. */
    private fun run(modelKey: String, variantId: String, crop: Bitmap): Head {
        val started = System.nanoTime()
        val spec = TfLiteModelManager.spec(context, modelKey, variantId)

        // The graph declares the size; the crop is squeezed to it. These heads
        // were trained on square crops resized the same way, so no letterbox.
        val resized = ImagePreprocessor.resizeBilinear(crop, spec.inputWidth, spec.inputHeight)
        val input = buildInput(spec, resized)
        if (resized !== crop) resized.recycle()

        val output = Array(1) { FloatArray(spec.outputLength) }
        spec.interpreter.run(input, output)
        val raw = output[0]

        val millis = (System.nanoTime() - started) / 1_000_000
        Log.i(TAG, "$modelKey/${variantId}: ${spec.inputWidth}x${spec.inputHeight} " +
            "-> [${raw.joinToString { "%.6f".format(it) }}] in $millis ms")

        return decode(modelKey, spec, raw, millis)
    }

    /**
     * Fill the input tensor for whatever the graph asks for.
     *
     * Both bundled heads take FLOAT32, but the buffer is built from
     * `spec.inputType` and the tensor's own quantisation parameters, so a fully
     * integer-quantised replacement drops in without a code change.
     */
    private fun buildInput(spec: TfLiteModelManager.Spec, image: Bitmap): ByteBuffer {
        val width = spec.inputWidth
        val height = spec.inputHeight
        val channels = spec.inputChannels
        val count = width * height * channels

        val pixels = IntArray(width * height)
        image.getPixels(pixels, 0, width, 0, 0, width, height)

        // Channel-last, RGB, matching the [1, H, W, C] the graphs declare.
        val values = FloatArray(count)
        var i = 0
        for (pixel in pixels) {
            values[i++] = ((pixel shr 16) and 0xFF).toFloat()
            if (channels > 1) values[i++] = ((pixel shr 8) and 0xFF).toFloat()
            if (channels > 2) values[i++] = (pixel and 0xFF).toFloat()
            for (c in 3 until channels) values[i++] = 0f
        }

        normalize(spec.config, values, channels)

        val bytesPerValue = spec.inputType.byteSize()
        val buffer = ByteBuffer.allocateDirect(count * bytesPerValue).order(ByteOrder.nativeOrder())
        when (spec.inputType) {
            DataType.FLOAT32 -> values.forEach { buffer.putFloat(it) }

            DataType.UINT8 -> {
                val scale = if (spec.inputQuantScale != 0f) spec.inputQuantScale else 1f
                values.forEach {
                    buffer.put(((it / scale) + spec.inputQuantZeroPoint).roundToInt().coerceIn(0, 255).toByte())
                }
            }

            DataType.INT8 -> {
                val scale = if (spec.inputQuantScale != 0f) spec.inputQuantScale else 1f
                values.forEach {
                    buffer.put(((it / scale) + spec.inputQuantZeroPoint).roundToInt().coerceIn(-128, 127).toByte())
                }
            }

            else -> throw IllegalStateException(
                "${spec.assetName} wants an unsupported input type ${spec.inputType}"
            )
        }
        buffer.rewind()
        return buffer
    }

    /** Apply the scheme `tflite_meta.json` names for this head, in place. */
    private fun normalize(
        config: TfLiteModelManager.ModelConfig,
        values: FloatArray,
        channels: Int,
    ) {
        when (config.normalization) {
            TfLiteModelManager.Normalization.NONE -> Unit

            TfLiteModelManager.Normalization.MEAN_STD -> {
                val scale = if (config.pixelScale != 0f) config.pixelScale else 1f
                for (i in values.indices) {
                    val c = i % channels
                    val mean = config.mean.getOrElse(c) { 0f }
                    val std = config.std.getOrElse(c) { 1f }
                    values[i] = ((values[i] / scale) - mean) / (if (std != 0f) std else 1f)
                }
            }

            // Port of tf.image.per_image_standardization: subtract this image's
            // own mean and divide by its own standard deviation, with the
            // 1/sqrt(n) floor that keeps a flat image from exploding.
            TfLiteModelManager.Normalization.PER_IMAGE_STANDARDIZATION -> {
                var sum = 0.0
                for (v in values) sum += v
                val mean = (sum / values.size).toFloat()

                var variance = 0.0
                for (v in values) {
                    val d = (v - mean).toDouble()
                    variance += d * d
                }
                val std = max(
                    sqrt(variance / values.size).toFloat(),
                    1f / sqrt(values.size.toFloat()),
                )
                for (i in values.indices) values[i] = (values[i] - mean) / std
            }
        }
    }

    private fun decode(
        modelKey: String,
        spec: TfLiteModelManager.Spec,
        raw: FloatArray,
        millis: Long,
    ): Head = when (val decode = spec.config.decode) {
        is TfLiteModelManager.Decode.LinearRegression -> {
            require(raw.isNotEmpty()) { "$modelKey produced an empty output" }
            val value = (raw[0] * decode.scale + decode.offset)
                .coerceIn(decode.clampMin, decode.clampMax)
            Head(label = modelKey, value = value, confidence = null, raw = raw, millis = millis)
        }

        is TfLiteModelManager.Decode.Classification -> {
            val probabilities = if (decode.alreadyNormalized && isProbabilityVector(raw)) {
                raw.copyOf()
            } else {
                softmax(raw)
            }

            var best = 0
            for (i in probabilities.indices) if (probabilities[i] > probabilities[best]) best = i

            if (decode.binWidth != null) {
                // A distribution over bins: the answer is its expectation.
                var expectation = 0f
                for (i in probabilities.indices) {
                    expectation += probabilities[i] * (decode.binOffset + i * decode.binWidth)
                }
                Head(modelKey, expectation, probabilities[best], raw, millis)
            } else {
                val label = decode.labels.getOrElse(best) { "class_$best" }
                Head(label, best.toFloat(), probabilities[best], raw, millis)
            }
        }
    }

    /** Cheap check that a head really did end in a softmax, before trusting it. */
    private fun isProbabilityVector(raw: FloatArray): Boolean {
        var sum = 0f
        for (v in raw) {
            if (v < -1e-3f || v > 1f + 1e-3f) return false
            sum += v
        }
        return kotlin.math.abs(sum - 1f) < 1e-2f
    }

    private fun softmax(raw: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in raw) if (v > maxLogit) maxLogit = v
        var sum = 0.0
        val out = FloatArray(raw.size)
        for (i in raw.indices) {
            val e = exp((raw[i] - maxLogit).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        if (sum > 0.0) for (i in out.indices) out[i] = (out[i] / sum).toFloat()
        return out
    }

    /**
     * Runs the bundled `assets/test.jpg` through the exact same path and logs
     * every intermediate number, so the pipeline can be checked without a face
     * in front of the camera:
     *
     *     adb shell am start -n <pkg>/.TfLiteActivity --ez run_self_test true
     *
     * The MiVOLO screen has the same hook; see CameraActivity.EXTRA_RUN_SELF_TEST.
     */
    fun runSelfTest(variantId: String): Result = try {
        val bitmap = context.assets.open(ModelManager.ASSET_SELF_TEST_IMAGE).use {
            android.graphics.BitmapFactory.decodeStream(it)
        } ?: return Result.Failure(
            "assets/${ModelManager.ASSET_SELF_TEST_IMAGE} could not be decoded", null
        )

        Log.i(TAG, "##### TFLITE SELF TEST on assets/${ModelManager.ASSET_SELF_TEST_IMAGE} #####")
        val result = analyze(bitmap, variantId)
        bitmap.recycle()
        result
    } catch (t: Throwable) {
        Log.e(TAG, "TFLite self test failed", t)
        Result.Failure(t.message ?: "self test failed", t)
    }

    private companion object {
        const val TAG = TfLiteModelManager.TAG

        /** The two keys the screen needs out of `tflite_meta.json`. */
        const val KEY_AGE = "age"
        const val KEY_GENDER = "gender"
    }
}
