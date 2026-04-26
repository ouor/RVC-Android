// Minimal on-device runtime for an AI-Hub-compiled QNN context binary
// (produced by tools/compile_synth_aihub.py for Hexagon V79). Only the
// operations needed by RvcSynthesizer are exposed:
//
//   * load a .bin from disk                (init)
//   * write into a named input tensor      (setInput)
//   * run the graph synchronously          (execute)
//   * read back a named output tensor      (getOutput)
//   * tear down                            (destroy)
//
// The runtime intentionally does not try to be a general QNN host —
// graph priorities, profiling, async dispatch, multiple graphs, etc.
// are all dropped. Adding them would not buy us anything for the
// single-graph synthesizer use case.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnTensor.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"

struct QnnTensorBinding {
    std::string name;
    std::vector<uint32_t> dims;
    Qnn_DataType_t dataType = QNN_DATATYPE_UNDEFINED;
    std::vector<uint8_t> buffer;  // owns the host-side memory the tensor points at
    Qnn_Tensor_t tensor{};         // points into buffer
};

class QnnSynthRuntime {
public:
    QnnSynthRuntime() = default;
    ~QnnSynthRuntime();

    // Returns false on any failure; check logcat under tag Rvc.QnnNative
    // for a specific reason. After failure, destroy() has already run.
    bool init(const std::string& binPath);
    bool execute();
    void destroy();

    // Input/output access by name, mirroring the ONNX input/output names.
    // Sizes must match the tensor's element count × element size; we
    // verify and reject mismatches loudly rather than silently truncating.
    bool setInput(const std::string& name, const void* src, size_t bytes);
    bool getOutput(const std::string& name, void* dst, size_t bytes) const;

    size_t numInputs() const { return inputs_.size(); }
    size_t numOutputs() const { return outputs_.size(); }

    // Diagnostics for logcat dumps.
    std::string describe() const;

private:
    // Library handles (dlopen targets are preloaded by Java's
    // OrtRuntime.ensureInitialized into the app's linker namespace).
    void* sysLib_ = nullptr;
    void* htpLib_ = nullptr;

    QNN_INTERFACE_VER_TYPE qnnApi_{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sysApi_{};

    Qnn_LogHandle_t logHandle_ = nullptr;
    Qnn_BackendHandle_t backendHandle_ = nullptr;
    Qnn_DeviceHandle_t deviceHandle_ = nullptr;
    Qnn_ContextHandle_t contextHandle_ = nullptr;
    Qnn_GraphHandle_t graphHandle_ = nullptr;

    std::vector<QnnTensorBinding> inputs_;
    std::vector<QnnTensorBinding> outputs_;

    // Plain-array views into inputs_/outputs_ for graphExecute().
    std::vector<Qnn_Tensor_t> inputTensors_;
    std::vector<Qnn_Tensor_t> outputTensors_;

    bool loadLibsAndInterfaces();
    bool setupBackendAndDevice();
    bool loadContextBinary(const std::string& binPath);
    bool buildBindings(const QnnSystemContext_BinaryInfo_t* binInfo);
    bool buildBindingsFromGraphInfo(const Qnn_Tensor_t* tensors,
                                    uint32_t count,
                                    std::vector<QnnTensorBinding>& out);
    QnnTensorBinding* findBinding(std::vector<QnnTensorBinding>& vec,
                                  const std::string& name);
    const QnnTensorBinding* findBinding(const std::vector<QnnTensorBinding>& vec,
                                        const std::string& name) const;
};
