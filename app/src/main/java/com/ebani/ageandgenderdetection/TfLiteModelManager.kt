package com.ebani.ageandgenderdetection

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Owns `assets/tflite_meta.json` and the TFLite interpreters built from
 * the `.tflite` files under `assets/tflite/`.
 *
 * The split of responsibility is deliberate and mirrors [ModelManager]:
 *
 *  - Anything the flatbuffer already states -- input height, width, channel
 *    count, tensor data type, quantisation scale/zero-point, output rank and
 *    length -- is read back off the [Interpreter] at load time and carried in
 *    [Spec]. None of it is written down in Kotlin.
 *  - Anything a TFLite graph physically cannot state -- how pixels are scaled
 *    before the first op, how the head turns into years, the class label order
 *    -- comes from `tflite_meta.json`.
 *
 * Consequence: dropping in a different age/gender pair needs a JSON edit, not a
 * code edit. Everything is loaded from local storage; the app still holds no
 * INTERNET permission.
 */
object TfLiteModelManager {

    const val TAG = "MiVOLO-TFLite"

    const val ASSET_META = "tflite_meta.json"

    /** How raw 0..255 pixels become model input. Selected by name from the JSON. */
    enum class Normalization {
        /** (pixel - batchMean) / max(batchStd, 1/sqrt(n)) -- tf.image.per_image_standardization. */
        PER_IMAGE_STANDARDIZATION,

        /** pixel / pixelScale, then (x - mean[c]) / std[c]. */
        MEAN_STD,

        /** pixel, untouched. */
        NONE;

        companion object {
            fun parse(raw: String): Normalization = when (raw.lowercase()) {
                "per_image_standardization", "standardize" -> PER_IMAGE_STANDARDIZATION
                "mean_std", "scale" -> MEAN_STD
                "none", "raw" -> NONE
                else -> throw IllegalArgumentException("unknown normalization \"$raw\" in $ASSET_META")
            }
        }
    }

    /** How the numbers a head emits turn into an answer. */
    sealed interface Decode {
        /** One unactivated scalar: `value * scale + offset`, clamped. */
        data class LinearRegression(
            val scale: Float,
            val offset: Float,
            val clampMin: Float,
            val clampMax: Float,
        ) : Decode

        /**
         * A vector over [labels]. A non-null [binWidth] means the head is a
         * distribution and the answer is its expectation over bin centres;
         * otherwise the answer is the argmax label.
         */
        data class Classification(
            val labels: List<String>,
            val alreadyNormalized: Boolean,
            val binWidth: Float?,
            val binOffset: Float,
        ) : Decode
    }

    /** Everything the JSON says about one head, before the graph is opened. */
    data class ModelConfig(
        val key: String,
        val fileStem: String,
        val normalization: Normalization,
        val pixelScale: Float,
        val mean: FloatArray,
        val std: FloatArray,
        val decode: Decode,
    )

    data class Variant(val id: String, val label: String, val suffix: String)

    /** Extra margin around the detector box, as a fraction of the box size. */
    data class FaceCrop(val marginX: Float, val marginY: Float)

    data class Meta(
        val assetDir: String,
        val defaultVariant: String,
        val variants: List<Variant>,
        val faceCrop: FaceCrop,
        val models: Map<String, ModelConfig>,
    ) {
        fun variant(id: String): Variant =
            variants.firstOrNull { it.id == id }
                ?: throw IllegalArgumentException("no variant \"$id\" in $ASSET_META")
    }

    /**
     * A loaded interpreter plus the shape and type facts read back out of it.
     *
     * [inputHeight]/[inputWidth]/[inputChannels] are whatever the graph declares,
     * so the 200x200 age head and the 128x128 gender head need no special-casing
     * anywhere upstream.
     */
    class Spec(
        val config: ModelConfig,
        val variantId: String,
        val assetName: String,
        val interpreter: Interpreter,
    ) {
        val inputShape: IntArray = interpreter.getInputTensor(0).shape()
        val outputShape: IntArray = interpreter.getOutputTensor(0).shape()

        val inputHeight = inputShape[1]
        val inputWidth = inputShape[2]
        val inputChannels = if (inputShape.size >= 4) inputShape[3] else 1
        val inputType: DataType = interpreter.getInputTensor(0).dataType()
        val inputQuantScale: Float = interpreter.getInputTensor(0).quantizationParams().scale
        val inputQuantZeroPoint: Int = interpreter.getInputTensor(0).quantizationParams().zeroPoint

        val outputType: DataType = interpreter.getOutputTensor(0).dataType()
        val outputQuantScale: Float = interpreter.getOutputTensor(0).quantizationParams().scale
        val outputQuantZeroPoint: Int = interpreter.getOutputTensor(0).quantizationParams().zeroPoint

        /** How many scalars the head emits: the product of the non-batch dims. */
        val outputLength: Int = outputShape.drop(1).fold(1) { a, b -> a * b }

        fun describe(): String =
            "$assetName in=[${inputShape.joinToString()}] $inputType " +
                "out=[${outputShape.joinToString()}] $outputType"

        fun close() = interpreter.close()
    }

    @Volatile private var meta: Meta? = null

    /** variantId -> (modelKey -> Spec) */
    private val specs = HashMap<String, MutableMap<String, Spec>>()

    fun meta(context: Context): Meta =
        meta ?: synchronized(this) { meta ?: loadMeta(context).also { meta = it } }

    /** The interpreter for one head of one variant, created on first use. */
    fun spec(context: Context, modelKey: String, variantId: String): Spec {
        val loaded = meta(context)
        val variant = loaded.variant(variantId)
        val config = loaded.models[modelKey]
            ?: throw IllegalArgumentException("no model \"$modelKey\" in $ASSET_META")
        synchronized(this) {
            val perVariant = specs.getOrPut(variantId) { HashMap() }
            perVariant[modelKey]?.let { return it }
            val created = create(context, config, variant, loaded.assetDir)
            perVariant[modelKey] = created
            return created
        }
    }

    /** Warm the default variant so the first capture is not charged for the load. */
    fun preload(context: Context) {
        val loaded = meta(context)
        Log.i(TAG, "----- tflite metadata ($ASSET_META) -----")
        Log.i(TAG, "  variants        : ${loaded.variants.map { it.id }} (default ${loaded.defaultVariant})")
        Log.i(TAG, "  face crop margin: x=${loaded.faceCrop.marginX} y=${loaded.faceCrop.marginY}")
        loaded.models.forEach { (key, config) ->
            Log.i(TAG, "  $key -> ${config.fileStem}* norm=${config.normalization} decode=${config.decode}")
        }
        loaded.models.keys.forEach { key ->
            val spec = spec(context, key, loaded.defaultVariant)
            Log.i(TAG, "  loaded $key: ${spec.describe()}")
        }
    }

    fun close() {
        synchronized(this) {
            specs.values.forEach { perVariant -> perVariant.values.forEach { runCatching { it.close() } } }
            specs.clear()
        }
    }

    /**
     * TFLite reads weights straight out of a [MappedByteBuffer], so the model is
     * memory-mapped from the APK rather than copied to filesDir and pulled onto
     * the Java heap. That needs the .tflite entries stored uncompressed -- see
     * the `noCompress` list in app/build.gradle.kts.
     */
    private fun create(
        context: Context,
        config: ModelConfig,
        variant: Variant,
        assetDir: String,
    ): Spec {
        val assetName = "$assetDir/${config.fileStem}${variant.suffix}.tflite"
        val started = System.nanoTime()
        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        }
        val interpreter = Interpreter(mapAsset(context, assetName), options)
        val spec = Spec(config, variant.id, assetName, interpreter)
        val millis = (System.nanoTime() - started) / 1_000_000
        Log.i(TAG, "${config.key}/${variant.id} interpreter ready in $millis ms -- ${spec.describe()}")
        return spec
    }

    private fun mapAsset(context: Context, assetName: String): MappedByteBuffer {
        val descriptor: AssetFileDescriptor = context.assets.openFd(assetName)
        descriptor.use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength,
                )
            }
        }
    }

    private fun loadMeta(context: Context): Meta {
        val text = context.assets.open(ASSET_META).bufferedReader().use { it.readText() }
        val json = JSONObject(text)

        val variantsJson = json.getJSONArray("variants")
        val variants = List(variantsJson.length()) { i ->
            val v = variantsJson.getJSONObject(i)
            Variant(
                id = v.getString("id"),
                label = v.optString("label", v.getString("id")),
                suffix = v.optString("suffix", ""),
            )
        }
        require(variants.isNotEmpty()) { "$ASSET_META declares no variants" }

        val cropJson = json.optJSONObject("face_crop")
        val faceCrop = FaceCrop(
            marginX = cropJson?.optDouble("margin_x", 0.0)?.toFloat() ?: 0f,
            marginY = cropJson?.optDouble("margin_y", 0.0)?.toFloat() ?: 0f,
        )

        val modelsJson = json.getJSONObject("models")
        val models = LinkedHashMap<String, ModelConfig>()
        for (key in modelsJson.keys()) {
            if (key.startsWith("_")) continue
            val m = modelsJson.getJSONObject(key)
            models[key] = ModelConfig(
                key = key,
                fileStem = m.getString("file_stem"),
                normalization = Normalization.parse(m.getString("normalization")),
                pixelScale = m.optDouble("pixel_scale", 1.0).toFloat(),
                mean = m.optJSONArray("mean").toFloatArrayOr(floatArrayOf(0f, 0f, 0f)),
                std = m.optJSONArray("std").toFloatArrayOr(floatArrayOf(1f, 1f, 1f)),
                decode = parseDecode(m.getJSONObject("decode")),
            )
        }
        require(models.isNotEmpty()) { "$ASSET_META declares no models" }

        val defaultVariant = json.optString("default_variant", variants.first().id)
        require(variants.any { it.id == defaultVariant }) {
            "default_variant \"$defaultVariant\" is not one of ${variants.map { it.id }}"
        }

        return Meta(
            assetDir = json.optString("asset_dir", "tflite"),
            defaultVariant = defaultVariant,
            variants = variants,
            faceCrop = faceCrop,
            models = models,
        )
    }

    private fun parseDecode(json: JSONObject): Decode = when (val kind = json.getString("kind")) {
        "linear_regression" -> Decode.LinearRegression(
            scale = json.optDouble("scale", 1.0).toFloat(),
            offset = json.optDouble("offset", 0.0).toFloat(),
            clampMin = json.optDouble("clamp_min", Double.NEGATIVE_INFINITY).toFloat(),
            clampMax = json.optDouble("clamp_max", Double.POSITIVE_INFINITY).toFloat(),
        )

        "classification" -> {
            val labelsJson = json.optJSONArray("labels")
            Decode.Classification(
                labels = if (labelsJson == null) emptyList()
                else List(labelsJson.length()) { labelsJson.getString(it) },
                alreadyNormalized = json.optBoolean("already_normalized", false),
                binWidth = if (json.has("bin_width")) json.getDouble("bin_width").toFloat() else null,
                binOffset = json.optDouble("bin_offset", 0.0).toFloat(),
            )
        }

        else -> throw IllegalArgumentException("unknown decode kind \"$kind\" in $ASSET_META")
    }

    private fun JSONArray?.toFloatArrayOr(fallback: FloatArray): FloatArray =
        if (this == null) fallback else FloatArray(length()) { getDouble(it).toFloat() }
}
