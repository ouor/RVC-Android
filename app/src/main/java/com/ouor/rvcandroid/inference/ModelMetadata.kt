package com.ouor.rvcandroid.inference

import ai.onnxruntime.OrtSession
import android.util.Log
import org.json.JSONObject

private const val TAG = "Rvc.Meta"

data class ModelMetadata(
    val samplingRate: Int,
    val f0: Boolean,
    val embChannels: Int,
    val embedder: String,
    val embOutputLayer: Int,
    val useFinalProj: Boolean,
    val modelType: String? = null,
    val version: String? = null,
    // Set by tools/export_static_synthesizer.py when the synth was exported
    // with fixed T (no dynamic_axes). Drives the static-T pipeline path.
    val staticT: Int? = null,
) {
    companion object {
        // voice-changer's export2onnx.py embeds these as a single JSON string under
        // the custom_metadata_props key "metadata"; we mirror its shape exactly so
        // a model exported by their tooling drops in unchanged.
        fun fromSession(session: OrtSession): ModelMetadata? {
            val raw = session.metadata.customMetadata["metadata"]
            if (raw == null) {
                Log.w(TAG, "no 'metadata' key in custom_metadata_props")
                return null
            }
            return parse(raw).also { Log.i(TAG, "parsed: $it") }
        }

        fun parse(json: String): ModelMetadata {
            val o = JSONObject(json)
            return ModelMetadata(
                samplingRate = o.getInt("samplingRate"),
                f0 = o.getBoolean("f0"),
                embChannels = o.getInt("embChannels"),
                embedder = o.getString("embedder"),
                embOutputLayer = o.getInt("embOutputLayer"),
                useFinalProj = o.optBoolean("useFinalProj", false),
                modelType = o.optString("modelType").ifEmpty { null },
                version = o.optString("version").ifEmpty { null },
                staticT = if (o.has("staticT")) o.getInt("staticT") else null,
            )
        }
    }
}
