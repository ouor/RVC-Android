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

#include <cstdint>
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

bool handleInfer(QnnSynthRuntime& rt, FILE* out, const wire::InferRequest& req) {
    // feats and rand_noise arrive as fp16 raw bits packed by the host
    // (android.util.Half); we hand the buffer straight to setInput, no
    // per-element conversion in this process.
    if (!rt.setInput("feats", req.feats.data(),
                     req.feats.size() * sizeof(uint16_t))) {
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
    if (!rt.setInput("rand_noise", req.noise.data(),
                     req.noise.size() * sizeof(uint16_t))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput rand_noise failed");
        return true;
    }

    if (!rt.execute()) {
        writeError(out, wire::STATUS_INFER, "graph execute failed");
        return true;
    }

    // Output is fp16 — copy raw bits into the response and let the host
    // unpack to fp32. AI Hub's qnn_context_binary compile renames the
    // original 'audio' output to e.g. 'output_0', so address by index.
    constexpr size_t AUDIO_ELEMENTS = 76800;  // 192 frames × 400 samples
    wire::InferResponse resp{};
    resp.status = wire::STATUS_OK;
    resp.audio.resize(AUDIO_ELEMENTS);
    if (!rt.getOutputByIndex(0, resp.audio.data(),
                             resp.audio.size() * sizeof(uint16_t))) {
        writeError(out, wire::STATUS_INFER, "getOutput[0] failed");
        return true;
    }
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
