package com.ebani.ageandgenderdetection

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * The whole on-device pipeline, end to end:
 *
 *     captured bitmap
 *        -> YuNet face detection            (FaceDetector)
 *        -> pick the subject, calibrate box  (FaceDetector)
 *        -> crop, letterbox, normalise       (ImagePreprocessor)
 *        -> MiVOLO ONNX                      (here)
 *        -> decode age + gender              (here)
 *
 * Decoding is a port of `mivolo/model/mi_volo.py :: fill_in_results`:
 *
 * ```python
 * age_output = output[:, 2]
 * gender_output = output[:, :2].softmax(-1)
 * gender_probs, gender_indx = gender_output.topk(1)
 * age = age_output[index].item()
 * age = age * (self.meta.max_age - self.meta.min_age) + self.meta.avg_age
 * gender = "male" if gender_indx[index].item() == 0 else "female"
 * ```
 *
 * The indices, the label order and the three age constants all come from
 * `model_meta.json`, which `scripts/export_model.py` reads out of the MiVOLO
 * checkpoint. Nothing here is guessed.
 */
class MiVOLOProcessor(private val context: Context) {

    private val meta = ModelManager.meta(context)
    private val session = ModelManager.mivoloSession(context)
    private val detector = FaceDetector(context)

    sealed interface Result {
        /** A face was found and analysed. */
        data class Success(
            val age: Float,
            val gender: String,
            val genderScore: Float,
            val faceBox: FaceDetector.Box,
            val faceCount: Int,
            val rawOutput: FloatArray,
            val detectMillis: Long,
            val inferenceMillis: Long,
            val totalMillis: Long,
        ) : Result {
            val ageRounded: Int get() = age.roundToInt()
            val genderDisplay: String get() = gender.replaceFirstChar { it.uppercase() }
        }

        /** No usable face. The UI shows the "capture again" path, never a crash. */
        data class NoFace(val reason: String) : Result

        /** Something went wrong loading or running the model. */
        data class Failure(val message: String, val cause: Throwable?) : Result
    }

    fun analyze(bitmap: Bitmap): Result {
        val startedAt = System.nanoTime()
        return try {
            Log.i(TAG, "===== analysis start =====")
            Log.i(TAG, "Input image size: ${bitmap.width}x${bitmap.height}")

            val detectStart = System.nanoTime()
            val faces = detector.detect(bitmap)
            val detectMillis = (System.nanoTime() - detectStart) / 1_000_000

            val primary = detector.selectPrimary(faces)
                ?: return Result.NoFace("no face above the detector score threshold").also {
                    Log.w(TAG, "No face detected in ${bitmap.width}x${bitmap.height} image")
                }

            Log.i(TAG, "Face detected: $primary  (selected from ${faces.size})")
            val calibrated = detector.calibrate(primary)
            Log.i(TAG, "Face box after MiVOLO calibration: $calibrated")

            val crop = ImagePreprocessor.cropClamped(
                bitmap,
                calibrated.x1.toInt(),
                calibrated.y1.toInt(),
                calibrated.x2.toInt(),
                calibrated.y2.toInt(),
            ) ?: return Result.NoFace("face box fell outside the image").also {
                Log.w(TAG, "Calibrated box did not intersect the image")
            }

            Log.i(TAG, "Face crop: ${crop.width}x${crop.height}")

            val tensor = ImagePreprocessor.prepare(crop, meta)
            crop.recycle()
            Log.i(TAG, "Model input shape: [1, ${meta.inChannels}, ${meta.inputSize}, ${meta.inputSize}]")
            Log.i(TAG, "Input tensor      : ${tensor.describe()}")

            Log.i(TAG, "Inference started")
            val inferenceStart = System.nanoTime()
            val output = OnnxTensor.createTensor(
                ModelManager.environment(), tensor.buffer, tensor.shape
            ).use { input ->
                session.run(mapOf(meta.inputName to input)).use { results ->
                    @Suppress("UNCHECKED_CAST")
                    (results[0].value as Array<FloatArray>)[0].copyOf()
                }
            }
            val inferenceMillis = (System.nanoTime() - inferenceStart) / 1_000_000
            Log.i(TAG, "Inference completed in ${inferenceMillis} ms")
            Log.i(TAG, "Raw model output  : [${output.joinToString { "%.6f".format(it) }}]")

            val (age, gender, genderScore) = decode(output)
            val totalMillis = (System.nanoTime() - startedAt) / 1_000_000

            Log.i(TAG, "Decoded age       : $age")
            Log.i(TAG, "Decoded gender    : $gender (${(genderScore * 100).roundToInt()}%)")
            Log.i(TAG, "===== analysis done in ${totalMillis} ms =====")

            Result.Success(
                age = age,
                gender = gender,
                genderScore = genderScore,
                faceBox = calibrated,
                faceCount = faces.size,
                rawOutput = output,
                detectMillis = detectMillis,
                inferenceMillis = inferenceMillis,
                totalMillis = totalMillis,
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Analysis failed", t)
            Result.Failure(t.message ?: t::class.java.simpleName, t)
        }
    }

    /** Port of mivolo/model/mi_volo.py :: fill_in_results. */
    private fun decode(output: FloatArray): Triple<Float, String, Float> {
        if (meta.onlyAge) {
            val age = output[0] * (meta.maxAge - meta.minAge) + meta.avgAge
            return Triple(age, "unknown", 0f)
        }

        val indices = meta.genderLogitIndices
        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in indices) if (output[i] > maxLogit) maxLogit = output[i]

        var sum = 0.0
        val probabilities = DoubleArray(indices.size)
        for ((slot, i) in indices.withIndex()) {
            val e = exp((output[i] - maxLogit).toDouble())
            probabilities[slot] = e
            sum += e
        }

        var best = 0
        for (slot in probabilities.indices) {
            probabilities[slot] /= sum
            if (probabilities[slot] > probabilities[best]) best = slot
        }

        val gender = meta.genderLabels.getOrElse(best) { "unknown" }
        val age = output[meta.ageOutputIndex] * (meta.maxAge - meta.minAge) + meta.avgAge
        return Triple(age, gender, probabilities[best].toFloat())
    }

    /**
     * Runs the bundled `assets/test.jpg` through the exact same path and logs a
     * tensor fingerprint, so the numbers can be diffed against
     * `python scripts/test_mivolo.py --image testdata/<same file> --json`.
     *
     * This is what makes requirement 7 (Python vs Android agreement) checkable
     * without needing to point a camera at anything.
     */
    fun runSelfTest(): Result {
        return try {
            val bitmap = context.assets.open(ModelManager.ASSET_SELF_TEST_IMAGE).use {
                android.graphics.BitmapFactory.decodeStream(it)
            } ?: return Result.Failure("assets/${ModelManager.ASSET_SELF_TEST_IMAGE} could not be decoded", null)

            Log.i(TAG, "##### SELF TEST on assets/${ModelManager.ASSET_SELF_TEST_IMAGE} #####")
            val result = analyze(bitmap)
            bitmap.recycle()
            result
        } catch (t: Throwable) {
            Log.e(TAG, "Self test failed", t)
            Result.Failure(t.message ?: "self test failed", t)
        }
    }

    private companion object {
        const val TAG = ModelManager.TAG
    }
}
