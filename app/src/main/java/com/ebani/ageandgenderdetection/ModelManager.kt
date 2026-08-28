package com.ebani.ageandgenderdetection

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Owns everything that comes out of `assets/`: the two ONNX graphs, the model
 * metadata written by `scripts/export_model.py`, and the face-box calibration
 * written by `scripts/calibrate_face_box.py`.
 *
 * No value used during inference is hardcoded in Kotlin. Input size, channel
 * order, normalisation constants and the age denormalisation constants are all
 * read from `model_meta.json`, which the export script derives from the MiVOLO
 * checkpoint itself. Swapping in a different MiVOLO checkpoint therefore needs
 * no Kotlin changes at all.
 *
 * Everything is loaded from local storage. Nothing is fetched at runtime; the
 * app holds no INTERNET permission.
 */
object ModelManager {

    const val TAG = "MiVOLO"

    const val ASSET_MODEL_META = "model_meta.json"
    const val ASSET_CALIBRATION = "face_box_calibration.json"
    const val ASSET_FACE_DETECTOR = "face_detection_yunet_2023mar.onnx"
    const val ASSET_SELF_TEST_IMAGE = "test.jpg"

    /**
     * Everything Android needs to know about the MiVOLO graph, straight from
     * `model_meta.json`.
     */
    data class ModelMeta(
        val modelFile: String,
        val sourceCheckpoint: String,
        val architecture: String,
        val inputName: String,
        val outputName: String,
        val inputSize: Int,
        val inChannels: Int,
        val numClasses: Int,
        val mean: FloatArray,
        val std: FloatArray,
        val pixelScale: Float,
        val onlyAge: Boolean,
        val minAge: Float,
        val maxAge: Float,
        val avgAge: Float,
        val genderLogitIndices: IntArray,
        val genderLabels: List<String>,
        val ageOutputIndex: Int,
    )

    /**
     * Maps a YuNet face box onto the box MiVOLO's own YOLOv8x detector would
     * have produced. See scripts/calibrate_face_box.py for how it is measured.
     */
    data class BoxCalibration(
        val scaleW: Float,
        val scaleH: Float,
        val shiftX: Float,
        val shiftY: Float,
        val samples: Int,
        val meanIouBefore: Float,
        val meanIouAfter: Float,
    ) {
        companion object {
            val IDENTITY = BoxCalibration(1f, 1f, 0f, 0f, 0, 0f, 0f)
        }
    }

    @Volatile private var environment: OrtEnvironment? = null
    @Volatile private var meta: ModelMeta? = null
    @Volatile private var calibration: BoxCalibration? = null
    @Volatile private var mivoloSession: OrtSession? = null
    @Volatile private var detectorSession: OrtSession? = null

    fun environment(): OrtEnvironment =
        environment ?: synchronized(this) {
            environment ?: OrtEnvironment.getEnvironment().also { environment = it }
        }

    fun meta(context: Context): ModelMeta =
        meta ?: synchronized(this) { meta ?: loadMeta(context).also { meta = it } }

    fun calibration(context: Context): BoxCalibration =
        calibration ?: synchronized(this) {
            calibration ?: loadCalibration(context).also { calibration = it }
        }

    fun mivoloSession(context: Context): OrtSession =
        mivoloSession ?: synchronized(this) {
            mivoloSession ?: createSession(context, meta(context).modelFile, "MiVOLO").also {
                mivoloSession = it
            }
        }

    fun detectorSession(context: Context): OrtSession =
        detectorSession ?: synchronized(this) {
            detectorSession ?: createSession(context, ASSET_FACE_DETECTOR, "YuNet").also {
                detectorSession = it
            }
        }

    /** Warm both graphs so the first capture is not charged for session creation. */
    fun preload(context: Context) {
        val loadedMeta = meta(context)
        Log.i(TAG, "----- model metadata -----")
        Log.i(TAG, "  checkpoint      : ${loadedMeta.sourceCheckpoint}")
        Log.i(TAG, "  architecture    : ${loadedMeta.architecture}")
        Log.i(TAG, "  input           : ${loadedMeta.inputName} " +
                "[1, ${loadedMeta.inChannels}, ${loadedMeta.inputSize}, ${loadedMeta.inputSize}] NCHW RGB")
        Log.i(TAG, "  mean            : ${loadedMeta.mean.joinToString()}")
        Log.i(TAG, "  std             : ${loadedMeta.std.joinToString()}")
        Log.i(TAG, "  age decode      : output[${loadedMeta.ageOutputIndex}] * " +
                "(${loadedMeta.maxAge} - ${loadedMeta.minAge}) + ${loadedMeta.avgAge}")
        Log.i(TAG, "  gender labels   : ${loadedMeta.genderLabels}")

        val cal = calibration(context)
        Log.i(TAG, "----- face box calibration -----")
        if (cal.samples == 0) {
            Log.w(TAG, "  identity transform (calibration not packaged)")
        } else {
            Log.i(TAG, "  scale=(${cal.scaleW}, ${cal.scaleH}) shift=(${cal.shiftX}, ${cal.shiftY})")
            Log.i(TAG, "  measured on ${cal.samples} face pairs, " +
                    "mean IoU ${cal.meanIouBefore} -> ${cal.meanIouAfter}")
        }

        detectorSession(context)
        Log.i(TAG, "YuNet face detector loaded ($ASSET_FACE_DETECTOR)")
        mivoloSession(context)
        Log.i(TAG, "MiVOLO model loaded (${loadedMeta.modelFile})")
    }

    fun close() {
        synchronized(this) {
            runCatching { mivoloSession?.close() }
            runCatching { detectorSession?.close() }
            mivoloSession = null
            detectorSession = null
        }
    }

    /**
     * ONNX Runtime can memory-map a session created from a file path, which
     * matters here: the MiVOLO graph is ~100 MB and reading it into a ByteArray
     * would put all of it on the Java heap. So the asset is copied into
     * `filesDir` once and the session is opened from that path afterwards.
     */
    private fun createSession(context: Context, assetName: String, label: String): OrtSession {
        val started = System.nanoTime()
        val file = materialise(context, assetName)
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(
                Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            )
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = environment().createSession(file.absolutePath, options)
        val millis = (System.nanoTime() - started) / 1_000_000
        Log.i(TAG, "$label session created in ${millis} ms from ${file.name} (${file.length()} bytes)")
        session.inputInfo.forEach { (name, info) -> Log.d(TAG, "  $label input  $name ${info.info}") }
        session.outputInfo.forEach { (name, info) -> Log.d(TAG, "  $label output $name ${info.info}") }
        return session
    }

    /** Copy an asset into filesDir once; reuse it when the size already matches. */
    private fun materialise(context: Context, assetName: String): File {
        val target = File(context.filesDir, assetName)
        val assetLength = context.assets.openFd(assetName).use { it.length }
        if (target.exists() && target.length() == assetLength) {
            return target
        }
        Log.i(TAG, "Extracting $assetName ($assetLength bytes) to ${target.absolutePath}")
        context.assets.open(assetName).use { input ->
            target.outputStream().use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE * 8) }
        }
        return target
    }

    private fun readAssetText(context: Context, assetName: String): String =
        context.assets.open(assetName).bufferedReader().use { it.readText() }

    private fun loadMeta(context: Context): ModelMeta {
        val json = JSONObject(readAssetText(context, ASSET_MODEL_META))
        return ModelMeta(
            modelFile = json.getString("model_file"),
            sourceCheckpoint = json.optString("source_checkpoint", "unknown"),
            architecture = json.optString("architecture", "unknown"),
            inputName = json.optString("input_name", "input"),
            outputName = json.optString("output_name", "output"),
            inputSize = json.getInt("input_size"),
            inChannels = json.getInt("in_chans"),
            numClasses = json.getInt("num_classes"),
            mean = json.getJSONArray("mean").toFloatArray(),
            std = json.getJSONArray("std").toFloatArray(),
            pixelScale = json.optDouble("pixel_scale", 255.0).toFloat(),
            onlyAge = json.optBoolean("only_age", false),
            minAge = json.getDouble("min_age").toFloat(),
            maxAge = json.getDouble("max_age").toFloat(),
            avgAge = json.getDouble("avg_age").toFloat(),
            genderLogitIndices = json.getJSONArray("gender_logit_indices").toIntArray(),
            genderLabels = json.getJSONArray("gender_labels").toStringList(),
            ageOutputIndex = json.getInt("age_output_index"),
        )
    }

    private fun loadCalibration(context: Context): BoxCalibration {
        val available = runCatching { context.assets.list("")?.contains(ASSET_CALIBRATION) == true }
            .getOrDefault(false)
        if (!available) {
            Log.w(TAG, "$ASSET_CALIBRATION missing from assets; using identity box transform")
            return BoxCalibration.IDENTITY
        }
        val json = JSONObject(readAssetText(context, ASSET_CALIBRATION))
        return BoxCalibration(
            scaleW = json.getDouble("scale_w").toFloat(),
            scaleH = json.getDouble("scale_h").toFloat(),
            shiftX = json.getDouble("shift_x").toFloat(),
            shiftY = json.getDouble("shift_y").toFloat(),
            samples = json.optInt("n_samples", 0),
            meanIouBefore = json.optDouble("mean_iou_before", 0.0).toFloat(),
            meanIouAfter = json.optDouble("mean_iou_after", 0.0).toFloat(),
        )
    }

    private fun JSONArray.toFloatArray() = FloatArray(length()) { getDouble(it).toFloat() }
    private fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }
    private fun JSONArray.toStringList() = List(length()) { getString(it) }
}
