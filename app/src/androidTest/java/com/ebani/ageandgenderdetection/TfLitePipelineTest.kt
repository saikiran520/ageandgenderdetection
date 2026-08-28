package com.ebani.ageandgenderdetection

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the bundled `assets/test.jpg` through the TFLite path on a real device.
 *
 * This covers the parts that a compile cannot: that both graphs are packaged
 * uncompressed and can be memory-mapped, that the shapes read off the flatbuffer
 * match what the preprocessor produces, and that the decoding in
 * `tflite_meta.json` yields a plausible age and one of the declared labels.
 */
@RunWith(AndroidJUnit4::class)
class TfLitePipelineTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun everyVariantProducesAnAgeAndAGender() {
        val processor = TfLiteProcessor(context)
        val variants = processor.variants()
        assertTrue("tflite_meta.json declares no variants", variants.isNotEmpty())

        for (variant in variants) {
            when (val result = processor.runSelfTest(variant.id)) {
                is TfLiteProcessor.Result.Success -> {
                    Log.i(
                        TfLiteModelManager.TAG,
                        "[test] ${variant.id}: age=${result.age} gender=${result.gender} " +
                            "score=${result.genderScore} in ${result.totalMillis} ms",
                    )
                    assertTrue(
                        "${variant.id} produced an implausible age ${result.age}",
                        result.age in 1f..116f,
                    )
                    assertTrue(
                        "${variant.id} produced an undeclared label ${result.gender}",
                        result.gender in setOf("male", "female"),
                    )
                    assertTrue(
                        "${variant.id} gender score out of range ${result.genderScore}",
                        result.genderScore in 0f..1f,
                    )
                }

                else -> throw AssertionError("${variant.id} self-test failed: $result")
            }
        }
    }

    /** The shapes must come off the flatbuffer, not out of Kotlin. */
    @Test
    fun specsAreReadFromTheGraphs() {
        val meta = TfLiteModelManager.meta(context)
        for (variant in meta.variants) {
            for (key in meta.models.keys) {
                val spec = TfLiteModelManager.spec(context, key, variant.id)
                Log.i(TfLiteModelManager.TAG, "[test] ${spec.describe()}")
                assertEquals("$key/${variant.id} is not [1, H, W, C]", 4, spec.inputShape.size)
                assertTrue("$key/${variant.id} has no input size", spec.inputWidth > 0)
                assertTrue("$key/${variant.id} has no output", spec.outputLength > 0)
            }
        }
    }
}
