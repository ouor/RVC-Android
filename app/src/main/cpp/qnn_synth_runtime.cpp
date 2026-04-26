#include "qnn_synth_runtime.h"

#include <android/log.h>
#include <dlfcn.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#define LOG_TAG "Rvc.QnnNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Element size for every Qnn_DataType we expect from the synth model.
// FLOAT_16 / FLOAT_32 / INT_32 cover what AI Hub emits after
// --truncate_64bit_io. UFIXED_POINT_8 / SFIXED_POINT_8 are listed in
// case a later quantised export is fed in.
size_t dtypeBytes(Qnn_DataType_t t) {
    switch (t) {
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
            return 2;
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            return 4;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return 8;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:
            return 1;
        default:
            return 0;
    }
}

const char* dtypeName(Qnn_DataType_t t) {
    switch (t) {
        case QNN_DATATYPE_FLOAT_16: return "fp16";
        case QNN_DATATYPE_FLOAT_32: return "fp32";
        case QNN_DATATYPE_INT_8:    return "i8";
        case QNN_DATATYPE_INT_16:   return "i16";
        case QNN_DATATYPE_INT_32:   return "i32";
        case QNN_DATATYPE_INT_64:   return "i64";
        case QNN_DATATYPE_UINT_8:   return "u8";
        case QNN_DATATYPE_UINT_16:  return "u16";
        case QNN_DATATYPE_UINT_32:  return "u32";
        case QNN_DATATYPE_UINT_64:  return "u64";
        case QNN_DATATYPE_BOOL_8:   return "bool";
        default:                    return "?";
    }
}

// Pull the v1 view out of a Qnn_Tensor_t regardless of the version field
// the binary happens to declare. v1 is a strict subset of every later
// version, and every read site below only touches v1 fields.
const Qnn_TensorV1_t& tv1(const Qnn_Tensor_t& t) {
    return t.v1;
}
Qnn_TensorV1_t& tv1(Qnn_Tensor_t& t) {
    return t.v1;
}

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        LOGE("readFile: cannot open %s", path.c_str());
        return false;
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char*>(out.data()), size)) {
        LOGE("readFile: short read for %s", path.c_str());
        return false;
    }
    return true;
}

void qnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t /*ts*/, va_list args) {
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, args);
    int p = ANDROID_LOG_INFO;
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:   p = ANDROID_LOG_ERROR; break;
        case QNN_LOG_LEVEL_WARN:    p = ANDROID_LOG_WARN;  break;
        case QNN_LOG_LEVEL_INFO:    p = ANDROID_LOG_INFO;  break;
        case QNN_LOG_LEVEL_DEBUG:   p = ANDROID_LOG_DEBUG; break;
        case QNN_LOG_LEVEL_VERBOSE: p = ANDROID_LOG_VERBOSE; break;
        default: break;
    }
    __android_log_print(p, "Rvc.QnnSdk", "%s", buf);
}

}  // namespace

QnnSynthRuntime::~QnnSynthRuntime() { destroy(); }

bool QnnSynthRuntime::init(const std::string& binPath) {
    if (!loadLibsAndInterfaces())   { destroy(); return false; }
    if (!setupBackendAndDevice())   { destroy(); return false; }
    if (!loadContextBinary(binPath)){ destroy(); return false; }
    LOGI("init: ready (%zu inputs, %zu outputs)", inputs_.size(), outputs_.size());
    return true;
}

bool QnnSynthRuntime::loadLibsAndInterfaces() {
    sysLib_ = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!sysLib_) {
        LOGE("dlopen libQnnSystem.so: %s", dlerror());
        return false;
    }
    htpLib_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!htpLib_) {
        LOGE("dlopen libQnnHtp.so: %s", dlerror());
        return false;
    }

    using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    using GetBackendProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

    auto sysGetProviders = reinterpret_cast<GetSysProviders>(
        dlsym(sysLib_, "QnnSystemInterface_getProviders"));
    if (!sysGetProviders) {
        LOGE("dlsym QnnSystemInterface_getProviders: %s", dlerror());
        return false;
    }
    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t sysProviderCount = 0;
    if (sysGetProviders(&sysProviders, &sysProviderCount) != QNN_SUCCESS || sysProviderCount == 0) {
        LOGE("QnnSystemInterface_getProviders: returned 0 providers");
        return false;
    }
    sysApi_ = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    auto backendGetProviders = reinterpret_cast<GetBackendProviders>(
        dlsym(htpLib_, "QnnInterface_getProviders"));
    if (!backendGetProviders) {
        LOGE("dlsym QnnInterface_getProviders: %s", dlerror());
        return false;
    }
    const QnnInterface_t** providers = nullptr;
    uint32_t providerCount = 0;
    if (backendGetProviders(&providers, &providerCount) != QNN_SUCCESS || providerCount == 0) {
        LOGE("QnnInterface_getProviders: returned 0 providers");
        return false;
    }
    qnnApi_ = providers[0]->QNN_INTERFACE_VER_NAME;
    LOGI("interfaces resolved: backend=%s sys=%s",
         providers[0]->providerName,
         sysProviders[0]->providerName);
    return true;
}

bool QnnSynthRuntime::setupBackendAndDevice() {
    if (qnnApi_.logCreate(qnnLogger, QNN_LOG_LEVEL_INFO, &logHandle_) != QNN_SUCCESS) {
        LOGE("logCreate failed");
        return false;
    }
    if (qnnApi_.backendCreate(logHandle_, nullptr, &backendHandle_) != QNN_SUCCESS) {
        LOGE("backendCreate failed");
        return false;
    }
    if (qnnApi_.deviceCreate(logHandle_, nullptr, &deviceHandle_) != QNN_SUCCESS) {
        LOGE("deviceCreate failed");
        return false;
    }
    return true;
}

bool QnnSynthRuntime::loadContextBinary(const std::string& binPath) {
    std::vector<uint8_t> binData;
    if (!readFile(binPath, binData)) return false;
    LOGI("context binary: %s (%.1f MiB)", binPath.c_str(),
         binData.size() / (1024.0 * 1024.0));

    QnnSystemContext_Handle_t sysCtx = nullptr;
    if (sysApi_.systemContextCreate(&sysCtx) != QNN_SUCCESS) {
        LOGE("systemContextCreate failed");
        return false;
    }

    const QnnSystemContext_BinaryInfo_t* binInfo = nullptr;
    Qnn_ContextBinarySize_t binInfoSize = 0;
    auto err = sysApi_.systemContextGetBinaryInfo(
        sysCtx, binData.data(), binData.size(), &binInfo, &binInfoSize);
    if (err != QNN_SUCCESS || binInfo == nullptr) {
        LOGE("systemContextGetBinaryInfo failed: 0x%llx",
             static_cast<unsigned long long>(err));
        sysApi_.systemContextFree(sysCtx);
        return false;
    }

    if (qnnApi_.contextCreateFromBinary(backendHandle_, deviceHandle_, nullptr,
                                        binData.data(), binData.size(),
                                        &contextHandle_, nullptr) != QNN_SUCCESS) {
        LOGE("contextCreateFromBinary failed");
        sysApi_.systemContextFree(sysCtx);
        return false;
    }

    if (!buildBindings(binInfo)) {
        sysApi_.systemContextFree(sysCtx);
        return false;
    }
    sysApi_.systemContextFree(sysCtx);
    return true;
}

bool QnnSynthRuntime::buildBindings(const QnnSystemContext_BinaryInfo_t* binInfo) {
    // The AI-Hub-compiled binary always carries a single graph for our
    // single-graph synthesizer. Versions v1/v2/v3 of the binary info
    // struct all expose graphs[0] as a QnnSystemContext_GraphInfo_t —
    // we just need to pull the v1 view out of that wrapper.
    const QnnSystemContext_GraphInfo_t* graphInfo = nullptr;
    switch (binInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            if (binInfo->contextBinaryInfoV1.numGraphs == 0) {
                LOGE("binary has no graphs");
                return false;
            }
            graphInfo = &binInfo->contextBinaryInfoV1.graphs[0];
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            if (binInfo->contextBinaryInfoV2.numGraphs == 0) {
                LOGE("binary has no graphs");
                return false;
            }
            graphInfo = &binInfo->contextBinaryInfoV2.graphs[0];
            break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            if (binInfo->contextBinaryInfoV3.numGraphs == 0) {
                LOGE("binary has no graphs");
                return false;
            }
            graphInfo = &binInfo->contextBinaryInfoV3.graphs[0];
            break;
        default:
            LOGE("unsupported binary info version: %d", binInfo->version);
            return false;
    }

    const char* graphName = nullptr;
    const Qnn_Tensor_t* gInputs = nullptr;
    const Qnn_Tensor_t* gOutputs = nullptr;
    uint32_t numIn = 0;
    uint32_t numOut = 0;
    switch (graphInfo->version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
            graphName = graphInfo->graphInfoV1.graphName;
            gInputs   = graphInfo->graphInfoV1.graphInputs;
            gOutputs  = graphInfo->graphInfoV1.graphOutputs;
            numIn     = graphInfo->graphInfoV1.numGraphInputs;
            numOut    = graphInfo->graphInfoV1.numGraphOutputs;
            break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
            graphName = graphInfo->graphInfoV2.graphName;
            gInputs   = graphInfo->graphInfoV2.graphInputs;
            gOutputs  = graphInfo->graphInfoV2.graphOutputs;
            numIn     = graphInfo->graphInfoV2.numGraphInputs;
            numOut    = graphInfo->graphInfoV2.numGraphOutputs;
            break;
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
            graphName = graphInfo->graphInfoV3.graphName;
            gInputs   = graphInfo->graphInfoV3.graphInputs;
            gOutputs  = graphInfo->graphInfoV3.graphOutputs;
            numIn     = graphInfo->graphInfoV3.numGraphInputs;
            numOut    = graphInfo->graphInfoV3.numGraphOutputs;
            break;
        default:
            LOGE("unsupported graph info version: %d", graphInfo->version);
            return false;
    }

    if (qnnApi_.graphRetrieve(contextHandle_, graphName, &graphHandle_) != QNN_SUCCESS) {
        LOGE("graphRetrieve(%s) failed", graphName ? graphName : "(null)");
        return false;
    }
    LOGI("graph '%s' retrieved (%u inputs, %u outputs)",
         graphName, numIn, numOut);

    if (!buildBindingsFromGraphInfo(gInputs, numIn, inputs_))   return false;
    if (!buildBindingsFromGraphInfo(gOutputs, numOut, outputs_)) return false;

    inputTensors_.clear();
    outputTensors_.clear();
    inputTensors_.reserve(inputs_.size());
    outputTensors_.reserve(outputs_.size());
    for (auto& b : inputs_)  inputTensors_.push_back(b.tensor);
    for (auto& b : outputs_) outputTensors_.push_back(b.tensor);
    return true;
}

bool QnnSynthRuntime::buildBindingsFromGraphInfo(const Qnn_Tensor_t* tensors,
                                                 uint32_t count,
                                                 std::vector<QnnTensorBinding>& out) {
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& src = tv1(tensors[i]);
        auto& b = out[i];
        b.name = src.name ? src.name : "";
        b.dataType = src.dataType;
        b.dims.assign(src.dimensions, src.dimensions + src.rank);

        size_t elements = 1;
        for (auto d : b.dims) elements *= d;
        const size_t elem = dtypeBytes(b.dataType);
        if (elem == 0) {
            LOGE("tensor[%u] '%s' has unsupported dtype %d",
                 i, b.name.c_str(), b.dataType);
            return false;
        }
        b.buffer.assign(elements * elem, 0);

        // Build the Qnn_Tensor_t the runtime will hand graphExecute. We
        // copy the descriptor and overwrite memType/clientBuf so QNN
        // reads/writes our allocated host buffer.
        b.tensor = tensors[i];
        auto& v1 = tv1(b.tensor);
        v1.memType = QNN_TENSORMEMTYPE_RAW;
        v1.clientBuf = Qnn_ClientBuffer_t{
            .data = b.buffer.data(),
            .dataSize = static_cast<uint32_t>(b.buffer.size()),
        };
        v1.dimensions = b.dims.data();
        v1.rank = static_cast<uint32_t>(b.dims.size());
        v1.name = b.name.c_str();
    }
    return true;
}

bool QnnSynthRuntime::execute() {
    if (!graphHandle_) {
        LOGE("execute: no graph loaded");
        return false;
    }
    // graphExecute reads the inputTensors_ array and writes into
    // outputTensors_'s clientBufs. We don't refresh inputTensors_ from
    // inputs_ here because setInput already wrote into the same backing
    // buffer the tensor descriptor points at.
    auto err = qnnApi_.graphExecute(
        graphHandle_,
        inputTensors_.data(), static_cast<uint32_t>(inputTensors_.size()),
        outputTensors_.data(), static_cast<uint32_t>(outputTensors_.size()),
        nullptr, nullptr);
    if (err != QNN_SUCCESS) {
        LOGE("graphExecute failed: 0x%llx", static_cast<unsigned long long>(err));
        return false;
    }
    return true;
}

void QnnSynthRuntime::destroy() {
    if (contextHandle_ && qnnApi_.contextFree) {
        qnnApi_.contextFree(contextHandle_, nullptr);
    }
    contextHandle_ = nullptr;
    graphHandle_ = nullptr;

    if (deviceHandle_ && qnnApi_.deviceFree) {
        qnnApi_.deviceFree(deviceHandle_);
    }
    deviceHandle_ = nullptr;

    if (backendHandle_ && qnnApi_.backendFree) {
        qnnApi_.backendFree(backendHandle_);
    }
    backendHandle_ = nullptr;

    if (logHandle_ && qnnApi_.logFree) {
        qnnApi_.logFree(logHandle_);
    }
    logHandle_ = nullptr;

    inputs_.clear();
    outputs_.clear();
    inputTensors_.clear();
    outputTensors_.clear();

    // Don't dlclose the libs — Java still holds them via System.loadLibrary
    // and we may construct another runtime in the same process.
    sysLib_ = nullptr;
    htpLib_ = nullptr;
}

bool QnnSynthRuntime::setInput(const std::string& name, const void* src, size_t bytes) {
    auto* b = findBinding(inputs_, name);
    if (!b) {
        LOGE("setInput: no input named '%s'", name.c_str());
        return false;
    }
    if (bytes != b->buffer.size()) {
        LOGE("setInput('%s'): %zu bytes given, %zu expected",
             name.c_str(), bytes, b->buffer.size());
        return false;
    }
    std::memcpy(b->buffer.data(), src, bytes);
    return true;
}

bool QnnSynthRuntime::getOutput(const std::string& name, void* dst, size_t bytes) const {
    const auto* b = findBinding(outputs_, name);
    if (!b) {
        LOGE("getOutput: no output named '%s'", name.c_str());
        return false;
    }
    if (bytes != b->buffer.size()) {
        LOGE("getOutput('%s'): %zu bytes asked for, %zu available",
             name.c_str(), bytes, b->buffer.size());
        return false;
    }
    std::memcpy(dst, b->buffer.data(), bytes);
    return true;
}

QnnTensorBinding* QnnSynthRuntime::findBinding(std::vector<QnnTensorBinding>& vec,
                                               const std::string& name) {
    for (auto& b : vec) if (b.name == name) return &b;
    return nullptr;
}

const QnnTensorBinding* QnnSynthRuntime::findBinding(
    const std::vector<QnnTensorBinding>& vec, const std::string& name) const {
    for (const auto& b : vec) if (b.name == name) return &b;
    return nullptr;
}

std::string QnnSynthRuntime::describe() const {
    std::ostringstream os;
    os << "inputs:";
    for (const auto& b : inputs_) {
        os << " " << b.name << ":" << dtypeName(b.dataType) << "[";
        for (size_t i = 0; i < b.dims.size(); ++i) {
            if (i) os << ",";
            os << b.dims[i];
        }
        os << "]";
    }
    os << " outputs:";
    for (const auto& b : outputs_) {
        os << " " << b.name << ":" << dtypeName(b.dataType) << "[";
        for (size_t i = 0; i < b.dims.size(); ++i) {
            if (i) os << ",";
            os << b.dims[i];
        }
        os << "]";
    }
    return os.str();
}
