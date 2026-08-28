package com.ebani.ageandgenderdetection

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.ebani.ageandgenderdetection.ui.theme.AgeandgenderdetectionTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import androidx.lifecycle.lifecycleScope

/**
 * Screen 1. A title and one button labelled "MiVOLO".
 *
 * The models are warmed up here on a background thread so that the first
 * capture on the camera screen does not pay for session creation.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        lifecycleScope.launch(Dispatchers.IO) {
            runCatching { ModelManager.preload(applicationContext) }
                .onFailure { Log.e(ModelManager.TAG, "Model preload failed", it) }
        }

        setContent {
            AgeandgenderdetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    HomeScreen(
                        contentPadding = innerPadding,
                        onOpenCamera = {
                            startActivity(Intent(this, CameraActivity::class.java))
                        },
                        onSelfTest = {
                            startActivity(
                                Intent(this, CameraActivity::class.java)
                                    .putExtra(CameraActivity.EXTRA_RUN_SELF_TEST, true)
                            )
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun HomeScreen(
    contentPadding: PaddingValues,
    onOpenCamera: () -> Unit,
    onSelfTest: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "MiVOLO Age & Gender POC",
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(8.dp))

        Text(
            text = "Runs entirely on this device. No network permission.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(40.dp))

        Button(
            onClick = onOpenCamera,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("MiVOLO")
        }

        Spacer(Modifier.height(16.dp))

        // Not part of the POC flow: this runs the bundled assets/test.jpg through
        // the identical pipeline and logs a tensor fingerprint, which is how the
        // Python-vs-Android comparison in the README is verified without a camera.
        // Safe to delete along with CameraActivity.EXTRA_RUN_SELF_TEST.
        TextButton(onClick = onSelfTest) {
            Text("Run bundled self-test (Python parity)")
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun HomeScreenPreview() {
    AgeandgenderdetectionTheme {
        HomeScreen(PaddingValues(0.dp), {}, {})
    }
}
