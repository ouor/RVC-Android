// RMVPE pitch-extractor sibling of rvc_synth_runner / rvc_hubert_runner.
// Same dlopen-based QNN host (qnn_synth_runtime), narrower wire
// protocol: two fp32 inputs (waveform, threshold) and one fp32 output
// (pitchf). The graph baked staticAudioLen=30720 → pitchf[1, 192],
// which lines up exactly with the synth's static T axis so the
// pipeline can run RMVPE per-synth-chunk just like HuBERT.
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

#define LOG_TAG "Rvc.QnnRmvRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct RmvpeRequest {
    uint32_t numSamples;
    std::vector<float> waveform;
    float threshold;                 // single fp32 scalar
};

struct RmvpeResponse {
    uint32_t status;
    uint32_t numFrames;              // T at 100 Hz = numSamples / 160
    std::vector<float> pitchf;       // fp32, length = numFrames
    std::string errMsg;
};

bool readRmvpeRequest(FILE* f, RmvpeRequest& req) {
    if (!wire::readU32(f, req.numSamples)) return false;
    if (!wire::readVec(f, req.waveform)) return false;
    if (!wire::readBytes(f, &req.threshold, sizeof(float))) return false;
    return true;
}

bool writeRmvpeResponse(FILE* f, const RmvpeResponse& resp) {
    if (!wire::writeU32(f, wire::MAGIC_RESP)) return false;
    if (!wire::writeU32(f, resp.status)) return false;
    if (!wire::writeU32(f, resp.numFrames)) return false;
    if (!wire::writeVec(f, resp.pitchf)) return false;
    if (!wire::writeStr(f, resp.errMsg)) return false;
    std::fflush(f);
    return true;
}

void writeError(FILE* out, uint32_t status, const std::string& msg) {
    RmvpeResponse resp{};
    resp.status = status;
    resp.errMsg = msg;
    writeRmvpeResponse(out, resp);
}

bool handleInfer(QnnSynthRuntime& rt, FILE* out, const RmvpeRequest& req) {
    if (!rt.setInput("waveform", req.waveform.data(),
                     req.waveform.size() * sizeof(float))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput waveform failed");
        return true;
    }
    if (!rt.setInput("threshold", &req.threshold, sizeof(float))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput threshold failed");
        return true;
    }
    if (!rt.execute()) {
        writeError(out, wire::STATUS_INFER, "graph execute failed");
        return true;
    }

    // Pull shape from the actual output binding rather than computing.
    // RMVPE on a 30720-sample input emits pitchf[1, 192]; we don't
    // hardcode that since other audio_samples values would change it.
    const auto* outBinding = rt.outputBinding(0);
    if (!outBinding) {
        writeError(out, wire::STATUS_INFER, "missing output binding");
        return true;
    }
    // Accept both [1, T] (rank 2) and [T] (rank 1) — voice-changer's
    // RMVPE export is rank-2 but AI Hub sometimes flattens trailing
    // ones. We treat the last dim as T regardless.
    if (outBinding->dims.empty()) {
        writeError(out, wire::STATUS_INFER, "output binding has no dims");
        return true;
    }
    const uint32_t numFrames = outBinding->dims.back();

    RmvpeResponse resp{};
    resp.status    = wire::STATUS_OK;
    resp.numFrames = numFrames;
    resp.pitchf.resize(numFrames);
    if (!rt.getOutputByIndex(0, resp.pitchf.data(),
                             resp.pitchf.size() * sizeof(float))) {
        writeError(out, wire::STATUS_INFER, "getOutput[0] failed");
        return true;
    }
    return writeRmvpeResponse(out, resp);
}

}  // namespace

int main(int argc, char** argv) {
    LOGI("rvc_rmvpe_runner starting (argc=%d)", argc);

    std::string binPath;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bin" && i + 1 < argc) binPath = argv[++i];
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
                RmvpeRequest req;
                if (!readRmvpeRequest(stdin, req)) {
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
