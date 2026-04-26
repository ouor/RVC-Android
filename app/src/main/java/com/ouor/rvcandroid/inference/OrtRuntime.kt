package com.ouor.rvcandroid.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import android.util.Log

private const val TAG = "Rvc.Ort"

object OrtRuntime {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    fun openSession(bytes: ByteArray): OrtSession {
        val t0 = System.currentTimeMillis()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(bytes, opts)
        val elapsed = System.currentTimeMillis() - t0
        Log.i(TAG, "openSession: ${bytes.size / (1024 * 1024)} MiB loaded in ${elapsed}ms")
        Log.d(TAG, "  inputs:  ${describe(session.inputInfo)}")
        Log.d(TAG, "  outputs: ${describe(session.outputInfo)}")
        return session
    }

    fun openSession(ctx: Context, uri: Uri): OrtSession {
        Log.i(TAG, "openSession: reading $uri")
        val bytes = ctx.contentResolver.openInputStream(uri)
            ?.use { it.readBytes() }
            ?: error("cannot open model uri: $uri")
        return openSession(bytes)
    }

    private fun describe(info: Map<String, NodeInfo>): String =
        info.entries.joinToString(prefix = "{", postfix = "}") { (name, ni) ->
            val v = ni.info
            if (v is TensorInfo) "$name:${v.type}${v.shape.contentToString()}"
            else "$name:$v"
        }
}
