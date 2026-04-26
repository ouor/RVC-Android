package com.ouor.rvcandroid.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.FloatBuffer
import java.nio.LongBuffer

fun OrtEnvironment.floatTensor(data: FloatArray, shape: LongArray): OnnxTensor =
    OnnxTensor.createTensor(this, FloatBuffer.wrap(data), shape)

fun OrtEnvironment.longTensor(data: LongArray, shape: LongArray): OnnxTensor =
    OnnxTensor.createTensor(this, LongBuffer.wrap(data), shape)

fun OnnxTensor.copyFloats(): FloatArray {
    val fb = floatBuffer ?: error("tensor is not float-typed")
    fb.rewind()
    val out = FloatArray(fb.remaining())
    fb.get(out)
    return out
}
