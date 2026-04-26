package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.util.Half
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

fun OrtEnvironment.floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
    OnnxTensor.createTensor(this, FloatBuffer.wrap(data), shape)

fun OrtEnvironment.longTensor(data: LongArray, shape: LongArray): OnnxTensor =
    OnnxTensor.createTensor(this, LongBuffer.wrap(data), shape)

// FLOAT16 path: voice-changer's is_half=True exports take feats as IEEE 754
// half-precision. android.util.Half (API 26+) provides exact bit conversions
// against the IEEE-754 reference, which matches what ORT's FLOAT16 type
// expects in a ShortBuffer.
fun OrtEnvironment.float16Tensor(data: FloatArray, shape: LongArray): OnnxTensor {
    val shorts = ShortArray(data.size) { Half.toHalf(data[it]) }
    return OnnxTensor.createTensor(this, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
}

fun OnnxTensor.copyFloats(): FloatArray {
    val fb = floatBuffer ?: error("tensor is not float-typed")
    fb.rewind()
    val out = FloatArray(fb.remaining())
    fb.get(out)
    return out
}

fun OnnxTensor.copyFloats16(): FloatArray {
    val sb = byteBuffer.asShortBuffer()
    sb.rewind()
    val out = FloatArray(sb.remaining())
    for (i in out.indices) out[i] = Half.toFloat(sb.get())
    return out
}
