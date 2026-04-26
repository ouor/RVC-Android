package com.ouor.rvcandroid.inference

// JNI surface for the native QNN runtime (libRvcQnn.so).
//
// The companion .so links against no QNN libraries at build time;
// libQnnSystem.so and libQnnHtp.so are preloaded into the app's linker
// namespace by OrtRuntime.ensureInitialized so the native side resolves
// QnnInterface_getProviders et al. via dlsym at runtime.
internal object QnnNative {
    init {
        System.loadLibrary("rvc_qnn")
    }

    @JvmStatic
    external fun nativeLoad(binPath: String): Long

    @JvmStatic
    external fun nativeInfer(
        handle: Long,
        feats: FloatArray,
        pLen: Int,
        pitch: IntArray,
        pitchf: FloatArray,
        sid: Int,
        randNoise: FloatArray,
        outAudioSize: Int,
    ): FloatArray?

    @JvmStatic
    external fun nativeRelease(handle: Long)
}
