package com.ouor.rvcandroid.inference

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File
import kotlin.math.absoluteValue

private const val TAG = "Rvc.Ort"

object OrtRuntime {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    fun openSession(file: File): OrtSession {
        val t0 = System.currentTimeMillis()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(file.absolutePath, opts)
        val elapsed = System.currentTimeMillis() - t0
        Log.i(
            TAG,
            "openSession: ${file.length() / (1024 * 1024)} MiB loaded from ${file.name} in ${elapsed}ms",
        )
        Log.d(TAG, "  inputs:  ${describe(session.inputInfo)}")
        Log.d(TAG, "  outputs: ${describe(session.outputInfo)}")
        return session
    }

    fun openSession(ctx: Context, uri: Uri): OrtSession {
        return openSession(ensureCachedFile(ctx, uri))
    }

    // SAF URIs cannot be opened as filesystem paths, but ORT's mmap-backed
    // createSession(filePath) can. Streaming each ONNX once into cacheDir
    // and feeding ORT the resulting path avoids loading 350+ MB models
    // through the Java heap, which previously OOM'd at the contentvec step.
    // Subsequent calls with the same URI reuse the cached copy.
    private fun ensureCachedFile(ctx: Context, uri: Uri): File {
        val expectedSize = ctx.contentResolver.openFileDescriptor(uri, "r")
            ?.use { it.statSize } ?: -1L
        val key = uri.toString().hashCode().toLong().absoluteValue
        val cached = File(ctx.cacheDir, "ort-$key-$expectedSize.onnx")
        if (cached.exists() && cached.length() == expectedSize && expectedSize > 0L) {
            Log.d(TAG, "ensureCachedFile: reusing ${cached.name}")
            return cached
        }
        ctx.cacheDir.listFiles { f -> f.name.startsWith("ort-$key-") }
            ?.forEach { if (it != cached) runCatching { it.delete() } }
        Log.i(
            TAG,
            "ensureCachedFile: copying $uri → ${cached.name} ($expectedSize bytes)",
        )
        val t0 = System.currentTimeMillis()
        ctx.contentResolver.openInputStream(uri)?.use { input ->
            cached.outputStream().use { input.copyTo(it) }
        } ?: error("cannot open model uri: $uri")
        Log.i(TAG, "ensureCachedFile: copied in ${System.currentTimeMillis() - t0}ms")
        return cached
    }

    private fun describe(info: Map<String, NodeInfo>): String =
        info.entries.joinToString(prefix = "{", postfix = "}") { (name, ni) ->
            val v = ni.info
            if (v is TensorInfo) "$name:${v.type}${v.shape.contentToString()}"
            else "$name:$v"
        }
}
