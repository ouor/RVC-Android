// Binary IPC protocol between the Android app process and the spawned
// rvc_synth_runner child process. The motivation for the spawn is the
// Hexagon HTP fastrpc unsigned-PD policy: untrusted_app SELinux context
// can't open an unsigned process domain on the DSP, but a binary
// spawned via ProcessBuilder transitions to a different domain where
// it can. The same QNN code therefore runs in a child process and we
// shovel tensors back and forth over stdin/stdout.
//
// Endianness: little-endian throughout. ARM64 Android is LE natively
// and Kotlin's ByteBuffer pins .order(LITTLE_ENDIAN) on the host side,
// so neither end performs byte-swapping.
//
// Framing:
//   - Each message starts with a 4-byte magic + 4-byte body length.
//   - Body is laid out as documented per command below.
//   - Strings: u32 length + raw UTF-8 bytes (no terminator).
//   - Arrays:  u32 element count + raw little-endian payload.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace wire {

constexpr uint32_t MAGIC_INFR = 0x52464E49u;  // 'I','N','F','R' in LE
constexpr uint32_t MAGIC_RESP = 0x50534552u;  // 'R','E','S','P' in LE
constexpr uint32_t MAGIC_REDY = 0x59444552u;  // 'R','E','D','Y' in LE
constexpr uint32_t MAGIC_QUIT = 0x54495551u;  // 'Q','U','I','T' in LE

constexpr uint32_t STATUS_OK       = 0;
constexpr uint32_t STATUS_BAD_REQ  = 1;
constexpr uint32_t STATUS_INFER    = 2;

// fp16 inputs/outputs travel as raw IEEE 754 binary16 bit patterns
// (uint16_t per element). The host (Kotlin) packs fp32→fp16 via
// android.util.Half before write and unpacks audio fp16→fp32 after
// read. This halves the per-call IPC payload (~720 KiB → ~370 KiB)
// and removes the runner's old f32→f16 staging vectors.
struct InferRequest {
    uint32_t framesT;
    uint32_t channels;
    std::vector<uint16_t> feats;    // fp16 bits; passed straight to setInput
    uint32_t pLen;
    std::vector<int32_t> pitch;
    std::vector<float> pitchf;      // fp32 — graph schema pins this input fp32
    uint32_t sid;
    std::vector<uint16_t> noise;    // fp16 bits; passed straight to setInput
};

struct InferResponse {
    uint32_t status;
    std::vector<uint16_t> audio;    // fp16 bits straight from getOutputByIndex
    std::string errMsg;
};

}  // namespace wire
