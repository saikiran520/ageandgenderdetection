package com.ebani.ageandgenderdetection

import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * YuNet face detection, run locally through ONNX Runtime.
 *
 * The decoding below is a direct port of `cv::FaceDetectorYNImpl::postProcess`
 * from opencv/modules/objdetect/src/face_detect.cpp. OpenCV's own class is not
 * used because the identical logic has to exist on the Python side too; keeping
 * both on raw ONNX Runtime means desktop and phone execute the same graph and
 * the same arithmetic, which is what makes the Python/Android comparison in the
 * README meaningful.
 *
 * Details taken from OpenCV rather than invented:
 *  - The blob is raw 0..255 float, **BGR**, NCHW. `cv::dnn::blobFromImage`
 *    defaults to scalefactor 1.0, no mean subtraction and swapRB = false.
 *  - score = sqrt(clamp(cls, 0, 1) * clamp(obj, 0, 1)).
 *  - cx = (col + bbox[0]) * stride, w = exp(bbox[2]) * stride, and the box is
 *    centre-form, so x1 = cx - w/2.
 *  - The image is padded bottom/right with zeros; padding never shifts the
 *    origin, so boxes map back to source pixels by a single divide.
 *
 * The bundled graph declares a fixed [1, 3, 640, 640] input, so the canvas is
 * always 640x640 and the three grids are always 80x80, 40x40 and 20x20.
 */
class FaceDetector(context: Context) {

    private val session = ModelManager.detectorSession(context)
    private val inputName = session.inputInfo.keys.first()
    private val calibration = ModelManager.calibration(context)

    data class Box(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
    ) {
        val width get() = x2 - x1
        val height get() = y2 - y1
        val area get() = max(0f, width) * max(0f, height)
        val cx get() = (x1 + x2) / 2f
        val cy get() = (y1 + y2) / 2f

        override fun toString(): String =
            "[${x1.roundToInt()},${y1.roundToInt()} -> ${x2.roundToInt()},${y2.roundToInt()}] " +
                "${width.roundToInt()}x${height.roundToInt()} score=${"%.3f".format(score)}"
    }

    /** Every face in the image, strongest first. */
    fun detect(bitmap: Bitmap): List<Box> {
        val started = System.nanoTime()

        // Only ever downscale: upsampling a small capture would blur it without
        // adding information.
        val scale = min(1.0f, CANVAS.toFloat() / max(bitmap.width, bitmap.height))
        val detW = if (scale < 1f) (bitmap.width * scale).roundToInt() else bitmap.width
        val detH = if (scale < 1f) (bitmap.height * scale).roundToInt() else bitmap.height

        val resized = if (scale < 1f) {
            ImagePreprocessor.resizeBilinear(bitmap, detW, detH)
        } else {
            bitmap
        }

        val plane = CANVAS * CANVAS
        val blob = FloatArray(3 * plane) // zero-initialised == the bottom/right padding
        val row = IntArray(detW)
        for (y in 0 until detH) {
            resized.getPixels(row, 0, detW, 0, y, detW, 1)
            val base = y * CANVAS
            for (x in 0 until detW) {
                val pixel = row[x]
                // BGR order, raw 0..255, to match cv::dnn::blobFromImage.
                blob[base + x] = (pixel and 0xFF).toFloat()
                blob[plane + base + x] = ((pixel shr 8) and 0xFF).toFloat()
                blob[2 * plane + base + x] = ((pixel shr 16) and 0xFF).toFloat()
            }
        }
        if (resized !== bitmap) resized.recycle()

        val raw = mutableListOf<Box>()
        OnnxTensor.createTensor(
            ModelManager.environment(),
            FloatBuffer.wrap(blob),
            longArrayOf(1, 3, CANVAS.toLong(), CANVAS.toLong()),
        ).use { tensor ->
            session.run(mapOf(inputName to tensor), OUTPUT_NAMES).use { results ->
                val outputs = OUTPUT_NAMES.associateWith { name ->
                    @Suppress("UNCHECKED_CAST")
                    (results.get(name).get().value as Array<Array<FloatArray>>)[0]
                }
                for (stride in STRIDES) {
                    decodeStride(stride, outputs, scale, raw)
                }
            }
        }

        val kept = nms(raw)
        val millis = (System.nanoTime() - started) / 1_000_000
        Log.i(TAG, "Face detection: ${kept.size} face(s) in ${millis} ms " +
                "(image ${bitmap.width}x${bitmap.height}, canvas ${CANVAS}x$CANVAS, scale ${"%.4f".format(scale)})")
        kept.forEachIndexed { index, box -> Log.d(TAG, "  face[$index] $box") }
        return kept
    }

    private fun decodeStride(
        stride: Int,
        outputs: Map<String, Array<FloatArray>>,
        scale: Float,
        into: MutableList<Box>,
    ) {
        val cls = outputs.getValue("cls_$stride")
        val obj = outputs.getValue("obj_$stride")
        val bbox = outputs.getValue("bbox_$stride")
        val cols = CANVAS / stride

        for (index in cls.indices) {
            val clsScore = cls[index][0].coerceIn(0f, 1f)
            val objScore = obj[index][0].coerceIn(0f, 1f)
            val score = sqrt(clsScore * objScore)
            if (score < SCORE_THRESHOLD) continue

            val r = index / cols
            val c = index % cols
            val prediction = bbox[index]
            val cx = (c + prediction[0]) * stride
            val cy = (r + prediction[1]) * stride
            val w = exp(prediction[2]) * stride
            val h = exp(prediction[3]) * stride
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f

            // Undo the detection-time downscale so boxes are in source pixels.
            into.add(
                Box(
                    x1 = x1 / scale,
                    y1 = y1 / scale,
                    x2 = (x1 + w) / scale,
                    y2 = (y1 + h) / scale,
                    score = score,
                )
            )
        }
    }

    private fun nms(boxes: List<Box>): List<Box> {
        val ordered = boxes.sortedByDescending { it.score }
        val kept = ArrayList<Box>()
        for (candidate in ordered) {
            if (kept.size >= TOP_K) break
            if (kept.none { iou(candidate, it) > NMS_THRESHOLD }) kept.add(candidate)
        }
        return kept
    }

    /**
     * The subject of the photo: the largest face.
     *
     * A POC capture has one intended subject, and in a hand-held portrait the
     * subject is nearer the camera than any bystander, so largest-area is more
     * stable than highest-confidence.
     */
    fun selectPrimary(boxes: List<Box>): Box? = boxes.maxByOrNull { it.area }

    /**
     * Map a YuNet box onto the box MiVOLO's own detector would have produced.
     *
     * MiVOLO crops the raw detector box with no margin and no alignment
     * (mivolo/structures.py :: crop_object), so the crop's tightness is part of
     * the model's input distribution. The four constants come from
     * scripts/calibrate_face_box.py, which measured them by running both
     * detectors over the same photographs.
     */
    fun calibrate(box: Box): Box {
        val cx = box.cx + calibration.shiftX * box.width
        val cy = box.cy + calibration.shiftY * box.height
        val w = box.width * calibration.scaleW
        val h = box.height * calibration.scaleH
        return Box(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f, box.score)
    }

    companion object {
        private const val TAG = ModelManager.TAG

        /** The bundled YuNet graph declares a fixed [1, 3, 640, 640] input. */
        const val CANVAS = 640
        private val STRIDES = intArrayOf(8, 16, 32)
        private const val SCORE_THRESHOLD = 0.6f
        private const val NMS_THRESHOLD = 0.3f
        private const val TOP_K = 50

        private val OUTPUT_NAMES = setOf(
            "cls_8", "cls_16", "cls_32",
            "obj_8", "obj_16", "obj_32",
            "bbox_8", "bbox_16", "bbox_32",
        )

        fun iou(a: Box, b: Box): Float {
            val ix1 = max(a.x1, b.x1)
            val iy1 = max(a.y1, b.y1)
            val ix2 = min(a.x2, b.x2)
            val iy2 = min(a.y2, b.y2)
            val intersection = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
            val union = a.area + b.area - intersection
            return if (union > 0f) intersection / union else 0f
        }
    }
}
