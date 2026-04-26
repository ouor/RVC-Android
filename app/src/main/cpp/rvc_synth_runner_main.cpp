// Entry point for the rvc_synth_runner child process. The Android app
// spawns this binary via ProcessBuilder; it inherits the app's uid but
// transitions to a different SELinux domain where the Hexagon DSP
// fastrpc driver allows opening an unsigned process domain — the
// blocker that kept the in-process JNI version (Phase η) from getting
// past QnnDevice_create.
//
// The child stays alive across many inferences. Spawn cost (load .bin,
// stand up QnnContext, retrieve graph) is paid once at startup; each
// chunk is just one round-trip on stdin/stdout.
#include "qnn_synth_runtime.h"
#include "wire_protocol.h"
#include "wire_protocol_io.h"

#include <android/log.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define LOG_TAG "Rvc.QnnRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

void writeError(FILE* out, uint32_t status, const std::string& msg) {
    wire::InferResponse resp{};
    resp.status = status;
    resp.errMsg = msg;
    wire::writeInferResponse(out, resp);
}

// Pack a float32 array into fp16 (uint16_t bit pattern). Same IEEE 754
// conversion the Phase η JNI used; lifted into the runner here so the
// host doesn't have to do FP packing twice (once for feats, once for
// noise).
uint16_t f32ToF16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (m + 0x1000u) >> 13);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 1u : 0u));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x1000u) >> 13));
}

float f16ToF32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp  = (h & 0x7c00u) >> 10;
    const uint32_t mant = (h & 0x03ffu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do { e++; m <<= 1; } while ((m & 0x400u) == 0);
            m &= 0x3ffu;
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

bool setFloatAsFp16(QnnSynthRuntime& rt, const std::string& name,
                    const std::vector<float>& src) {
    std::vector<uint16_t> half(src.size());
    for (size_t i = 0; i < src.size(); ++i) half[i] = f32ToF16(src[i]);
    return rt.setInput(name, half.data(), half.size() * sizeof(uint16_t));
}

bool handleInfer(QnnSynthRuntime& rt, FILE* out, const wire::InferRequest& req) {
    if (!setFloatAsFp16(rt, "feats", req.feats)) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput feats failed");
        return true;
    }
    {
        const int32_t v = static_cast<int32_t>(req.pLen);
        if (!rt.setInput("p_len", &v, sizeof(v))) {
            writeError(out, wire::STATUS_BAD_REQ, "setInput p_len failed");
            return true;
        }
    }
    if (!rt.setInput("pitch", req.pitch.data(),
                     req.pitch.size() * sizeof(int32_t))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput pitch failed");
        return true;
    }
    if (!rt.setInput("pitchf", req.pitchf.data(),
                     req.pitchf.size() * sizeof(float))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput pitchf failed");
        return true;
    }
    {
        const int32_t v = static_cast<int32_t>(req.sid);
        if (!rt.setInput("sid", &v, sizeof(v))) {
            writeError(out, wire::STATUS_BAD_REQ, "setInput sid failed");
            return true;
        }
    }
    if (!setFloatAsFp16(rt, "rand_noise", req.noise)) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput rand_noise failed");
        return true;
    }

    if (!rt.execute()) {
        writeError(out, wire::STATUS_INFER, "graph execute failed");
        return true;
    }

    // The synth's audio output is fp16 → unpack to fp32 for the host.
    // Output element count is fixed by the binary; address by index 0
    // since AI Hub's qnn_context_binary compile renames the original
    // 'audio' output to e.g. 'output_0'.
    constexpr size_t AUDIO_ELEMENTS = 76800;  // 192 frames × 400 samples
    std::vector<uint16_t> audio16(AUDIO_ELEMENTS);
    if (!rt.getOutputByIndex(0, audio16.data(),
                             audio16.size() * sizeof(uint16_t))) {
        writeError(out, wire::STATUS_INFER, "getOutput[0] failed");
        return true;
    }

    wire::InferResponse resp{};
    resp.status = wire::STATUS_OK;
    resp.audio.resize(audio16.size());
    for (size_t i = 0; i < audio16.size(); ++i) resp.audio[i] = f16ToF32(audio16[i]);
    return wire::writeInferResponse(out, resp);
}

}  // namespace

int main(int argc, char** argv) {
    LOGI("rvc_synth_runner starting (argc=%d)", argc);

    std::string binPath;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bin" && i + 1 < argc) {
            binPath = argv[++i];
        }
    }
    if (binPath.empty()) {
        LOGE("missing --bin <path>");
        std::fprintf(stderr, "missing --bin <path>\n");
        return 1;
    }

    QnnSynthRuntime rt;
    if (!rt.init(binPath)) {
        LOGE("runtime init failed");
        std::fprintf(stderr, "runtime init failed\n");
        return 1;
    }
    LOGI("runtime ready: %s", rt.describe().c_str());

    if (!wire::writeReady(stdout)) {
        LOGE("could not signal READY");
        return 1;
    }

    while (true) {
        const uint32_t magic = wire::readMagic(stdin);
        if (magic == 0) {
            LOGI("stdin EOF, exiting");
            break;
        }
        switch (magic) {
            case wire::MAGIC_INFR: {
                wire::InferRequest req;
                if (!wire::readInferRequest(stdin, req)) {
                    LOGE("malformed INFR body, exiting");
                    return 1;
                }
                if (!handleInfer(rt, stdout, req)) {
                    LOGE("response write failed, exiting");
                    return 1;
                }
                break;
            }
            case wire::MAGIC_QUIT: {
                LOGI("received QUIT");
                return 0;
            }
            default: {
                LOGE("unknown magic 0x%08x, exiting", magic);
                return 1;
            }
        }
    }
    return 0;
}
