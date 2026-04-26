// JNI bridge for the on-device QNN synthesizer runtime. The Kotlin side
// (com.ouor.rvcandroid.inference.QnnNative) holds an opaque jlong handle
// returned by nativeLoad; nativeInfer copies the six input arrays into
// the runtime's tensor buffers, executes the graph, and returns the
// FP16 audio output decoded to FloatArray.
#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>

#include "qnn_synth_runtime.h"

#define LOG_TAG "Rvc.QnnNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// IEEE 754 fp16 ↔ fp32. Matches what android.util.Half does on the
// Kotlin side; doing it in C++ avoids an extra Java-side conversion
// pass over a 76 800-element output every chunk.
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    const uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | (m + 0x1000) >> 13);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00 | (mant ? 1 : 0));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x1000) >> 13));
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h & 0x7c00u) >> 10;
    const uint32_t mant = (h & 0x03ffu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // subnormal — promote
            int e = -1;
            uint32_t m = mant;
            do { e++; m <<= 1; } while ((m & 0x400) == 0);
            m &= 0x3ff;
            out = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// Helpers to push/pull JNI arrays into a runtime input/output by name.
// All three primitive jarray accessors copy by default on Android, so we
// use Get*ArrayElements + JNI_ABORT to skip the redundant write-back.
template <typename T>
const T* lockArray(JNIEnv* env, jarray arr) {
    if (!arr) return nullptr;
    return reinterpret_cast<const T*>(env->GetPrimitiveArrayCritical(arr, nullptr));
}
void releaseArray(JNIEnv* env, jarray arr, const void* p) {
    if (!arr || !p) return;
    env->ReleasePrimitiveArrayCritical(arr, const_cast<void*>(p), JNI_ABORT);
}

bool setFloatAsFp16(QnnSynthRuntime& rt, const std::string& name,
                    const float* data, size_t count) {
    std::vector<uint16_t> half(count);
    for (size_t i = 0; i < count; ++i) half[i] = f32_to_f16(data[i]);
    return rt.setInput(name, half.data(), half.size() * sizeof(uint16_t));
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_ouor_rvcandroid_inference_QnnNative_nativeLoad(
    JNIEnv* env, jclass /*clazz*/, jstring binPath) {
    const char* path = env->GetStringUTFChars(binPath, nullptr);
    LOGI("nativeLoad: %s", path);
    auto* rt = new QnnSynthRuntime();
    if (!rt->init(path)) {
        delete rt;
        env->ReleaseStringUTFChars(binPath, path);
        return 0;
    }
    LOGI("nativeLoad: %s", rt->describe().c_str());
    env->ReleaseStringUTFChars(binPath, path);
    return reinterpret_cast<jlong>(rt);
}

JNIEXPORT void JNICALL
Java_com_ouor_rvcandroid_inference_QnnNative_nativeRelease(
    JNIEnv* /*env*/, jclass /*clazz*/, jlong handle) {
    LOGI("nativeRelease: handle=0x%llx", static_cast<long long>(handle));
    auto* rt = reinterpret_cast<QnnSynthRuntime*>(handle);
    delete rt;
}

JNIEXPORT jfloatArray JNICALL
Java_com_ouor_rvcandroid_inference_QnnNative_nativeInfer(
    JNIEnv* env, jclass /*clazz*/, jlong handle,
    jfloatArray feats,            // [1, T, C] FP32 from Kotlin → packed to FP16
    jint pLen,
    jintArray pitch,              // [1, T] int32
    jfloatArray pitchf,           // [1, T] FP32
    jint sid,
    jfloatArray randNoise,        // [1, IC, T] FP32 → packed to FP16
    jint outAudioSize) {

    auto* rt = reinterpret_cast<QnnSynthRuntime*>(handle);
    if (!rt) {
        LOGE("nativeInfer: null handle");
        return nullptr;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // ---- inputs ----
    {
        const auto* p = lockArray<float>(env, feats);
        const jsize n = env->GetArrayLength(feats);
        const bool ok = setFloatAsFp16(*rt, "feats", p, n);
        releaseArray(env, feats, p);
        if (!ok) return nullptr;
    }
    {
        const int32_t v = static_cast<int32_t>(pLen);
        if (!rt->setInput("p_len", &v, sizeof(v))) return nullptr;
    }
    {
        const auto* p = lockArray<int32_t>(env, pitch);
        const jsize n = env->GetArrayLength(pitch);
        const bool ok = rt->setInput("pitch", p, n * sizeof(int32_t));
        releaseArray(env, pitch, p);
        if (!ok) return nullptr;
    }
    {
        const auto* p = lockArray<float>(env, pitchf);
        const jsize n = env->GetArrayLength(pitchf);
        const bool ok = rt->setInput("pitchf", p, n * sizeof(float));
        releaseArray(env, pitchf, p);
        if (!ok) return nullptr;
    }
    {
        const int32_t v = static_cast<int32_t>(sid);
        if (!rt->setInput("sid", &v, sizeof(v))) return nullptr;
    }
    {
        const auto* p = lockArray<float>(env, randNoise);
        const jsize n = env->GetArrayLength(randNoise);
        const bool ok = setFloatAsFp16(*rt, "rand_noise", p, n);
        releaseArray(env, randNoise, p);
        if (!ok) return nullptr;
    }

    const auto tBuilt = std::chrono::steady_clock::now();
    if (!rt->execute()) return nullptr;
    const auto tRan = std::chrono::steady_clock::now();

    // ---- output: audio fp16 → FloatArray ----
    std::vector<uint16_t> audio16(static_cast<size_t>(outAudioSize));
    if (!rt->getOutput("audio", audio16.data(),
                        audio16.size() * sizeof(uint16_t))) return nullptr;
    jfloatArray out = env->NewFloatArray(outAudioSize);
    if (!out) return nullptr;
    std::vector<float> audio32(audio16.size());
    for (size_t i = 0; i < audio16.size(); ++i) audio32[i] = f16_to_f32(audio16[i]);
    env->SetFloatArrayRegion(out, 0, outAudioSize, audio32.data());

    const auto tDone = std::chrono::steady_clock::now();
    using ms = std::chrono::milliseconds;
    LOGI("nativeInfer: build=%lldms run=%lldms decode=%lldms",
         std::chrono::duration_cast<ms>(tBuilt - t0).count(),
         std::chrono::duration_cast<ms>(tRan - tBuilt).count(),
         std::chrono::duration_cast<ms>(tDone - tRan).count());
    return out;
}

}  // extern "C"
