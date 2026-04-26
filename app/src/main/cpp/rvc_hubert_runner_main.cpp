// HuBERT (ContentVec) sibling of rvc_synth_runner. Shares the QNN host
// glue in qnn_synth_runtime.* but speaks a much narrower wire protocol:
// one fp32 audio chunk in (1.92 s @ 16 kHz = 30720 samples) and one
// fp16 feature buffer out ([1, 96, 768], where 96 = 30720 / hop=320 and
// 768 is the v2 ContentVec embedding width).
//
// We spawn a SEPARATE PIE process for HuBERT instead of multiplexing
// over the synth runner because each child gets its own fastrpc
// process domain on the DSP — keeping them isolated also keeps QNN's
// graph state simple (one Qnn_Context per runner).
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

#define LOG_TAG "Rvc.QnnHubRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// HuBERT-specific INFR/RESP body shapes. Magic numbers are reused from
// wire_protocol.h (INFR/RESP/REDY/QUIT) so the runner-side magic table
// stays consistent across both child types.
struct HubertRequest {
    uint32_t numSamples;
    std::vector<float> audio;       // fp32, length = numSamples
};

struct HubertResponse {
    uint32_t status;
    uint32_t numFrames;             // ContentVec convs lose 1 frame to receptive
                                    // field; AI Hub's static compile of a 5120-
                                    // sample input yields 15 not 16 frames.
    uint32_t channels;              // 768 for v2
    std::vector<float> features;    // fp32 (matches the .bin's output dtype —
                                    // AI Hub kept fp32 IO on this graph even
                                    // though dispatch is fp16). Length =
                                    // numFrames × channels.
    std::string errMsg;
};

bool readHubertRequest(FILE* f, HubertRequest& req) {
    if (!wire::readU32(f, req.numSamples)) return false;
    if (!wire::readVec(f, req.audio)) return false;
    return true;
}

bool writeHubertResponse(FILE* f, const HubertResponse& resp) {
    if (!wire::writeU32(f, wire::MAGIC_RESP)) return false;
    if (!wire::writeU32(f, resp.status)) return false;
    if (!wire::writeU32(f, resp.numFrames)) return false;
    if (!wire::writeU32(f, resp.channels)) return false;
    if (!wire::writeVec(f, resp.features)) return false;
    if (!wire::writeStr(f, resp.errMsg)) return false;
    std::fflush(f);
    return true;
}

void writeError(FILE* out, uint32_t status, const std::string& msg) {
    HubertResponse resp{};
    resp.status = status;
    resp.errMsg = msg;
    writeHubertResponse(out, resp);
}

bool handleInfer(QnnSynthRuntime& rt, FILE* out, const HubertRequest& req) {
    // Audio input stays fp32 — voice-changer's ContentVec ONNX exposes
    // `audio` as float, and AI Hub preserves input dtypes when
    // compiling. The DSP converts to fp16 internally for dispatch.
    if (!rt.setInput("audio", req.audio.data(),
                     req.audio.size() * sizeof(float))) {
        writeError(out, wire::STATUS_BAD_REQ, "setInput audio failed");
        return true;
    }
    if (!rt.execute()) {
        writeError(out, wire::STATUS_INFER, "graph execute failed");
        return true;
    }

    // Read shape from the actual binding rather than computing from
    // numSamples — ContentVec's stride-5/3 conv stack drops 1 output
    // frame relative to the naive samples/320, and AI Hub may emit
    // fp16 on some graphs and fp32 on others. The schema printed at
    // init() is the source of truth.
    const auto* outBinding = rt.outputBinding(0);
    if (!outBinding || outBinding->dims.size() != 3) {
        writeError(out, wire::STATUS_INFER, "output binding shape != [1, T, C]");
        return true;
    }
    const uint32_t numFrames = outBinding->dims[1];
    const uint32_t channels  = outBinding->dims[2];

    HubertResponse resp{};
    resp.status    = wire::STATUS_OK;
    resp.numFrames = numFrames;
    resp.channels  = channels;
    resp.features.resize(static_cast<size_t>(numFrames) * channels);
    if (!rt.getOutputByIndex(0, resp.features.data(),
                             resp.features.size() * sizeof(float))) {
        writeError(out, wire::STATUS_INFER, "getOutput[0] failed");
        return true;
    }
    return writeHubertResponse(out, resp);
}

}  // namespace

int main(int argc, char** argv) {
    LOGI("rvc_hubert_runner starting (argc=%d)", argc);

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

    // QnnSynthRuntime is a misnomer post-Phase ξ — the class only knows
    // how to load a QNN context binary and read/write tensors by name,
    // it makes no assumptions about which model. Reusing it here keeps
    // the QNN init path in exactly one place.
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
                HubertRequest req;
                if (!readHubertRequest(stdin, req)) {
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
