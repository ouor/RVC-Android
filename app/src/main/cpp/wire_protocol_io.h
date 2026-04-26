// stdin/stdout serializers used by the runner main loop. Same format
// as documented in wire_protocol.h. Reads are blocking; on EOF or
// short read the helpers return false and the runner exits.
#pragma once

#include "wire_protocol.h"

#include <cstdio>
#include <vector>

namespace wire {

inline bool readBytes(FILE* f, void* dst, size_t n) {
    return std::fread(dst, 1, n, f) == n;
}

inline bool writeBytes(FILE* f, const void* src, size_t n) {
    return std::fwrite(src, 1, n, f) == n;
}

inline bool readU32(FILE* f, uint32_t& out) {
    return readBytes(f, &out, sizeof(out));
}

inline bool writeU32(FILE* f, uint32_t v) {
    return writeBytes(f, &v, sizeof(v));
}

template <typename T>
inline bool readVec(FILE* f, std::vector<T>& out) {
    uint32_t count;
    if (!readU32(f, count)) return false;
    out.resize(count);
    if (count == 0) return true;
    return readBytes(f, out.data(), count * sizeof(T));
}

template <typename T>
inline bool writeVec(FILE* f, const std::vector<T>& v) {
    if (!writeU32(f, static_cast<uint32_t>(v.size()))) return false;
    if (v.empty()) return true;
    return writeBytes(f, v.data(), v.size() * sizeof(T));
}

inline bool writeStr(FILE* f, const std::string& s) {
    if (!writeU32(f, static_cast<uint32_t>(s.size()))) return false;
    if (s.empty()) return true;
    return writeBytes(f, s.data(), s.size());
}

// Returns the next message magic, or 0 on EOF. Does NOT read the body
// — caller switches on magic and reads the appropriate body shape.
inline uint32_t readMagic(FILE* f) {
    uint32_t magic;
    if (!readU32(f, magic)) return 0;
    return magic;
}

inline bool readInferRequest(FILE* f, InferRequest& req) {
    if (!readU32(f, req.framesT))  return false;
    if (!readU32(f, req.channels)) return false;
    if (!readVec(f, req.feats))    return false;
    if (!readU32(f, req.pLen))     return false;
    if (!readVec(f, req.pitch))    return false;
    if (!readVec(f, req.pitchf))   return false;
    if (!readU32(f, req.sid))      return false;
    if (!readVec(f, req.noise))    return false;
    return true;
}

inline bool writeReady(FILE* f) {
    if (!writeU32(f, MAGIC_REDY)) return false;
    std::fflush(f);
    return true;
}

inline bool writeInferResponse(FILE* f, const InferResponse& resp) {
    if (!writeU32(f, MAGIC_RESP)) return false;
    if (!writeU32(f, resp.status)) return false;
    if (!writeVec(f, resp.audio)) return false;
    if (!writeStr(f, resp.errMsg)) return false;
    std::fflush(f);
    return true;
}

}  // namespace wire
