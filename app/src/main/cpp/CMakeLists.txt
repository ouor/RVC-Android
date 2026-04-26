cmake_minimum_required(VERSION 3.18)
project(rvc_qnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# QNN SDK headers are populated by tools/setup_qnn_libs.sh from a local
# install (see app/src/main/cpp/3rdparty/qnn/include/QNN).  Functions
# from libQnnSystem.so / libQnnHtp.so are not linked statically — Java
# preloads the .so files from jniLibs/arm64-v8a/ and our C++ resolves
# symbols via dlopen at runtime, mirroring how QNN's own SampleApp
# links against the SDK.
set(QNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/qnn/include)

if(NOT EXISTS ${QNN_INCLUDE_DIR}/QNN/QnnInterface.h)
    message(FATAL_ERROR
        "QNN headers not found at ${QNN_INCLUDE_DIR}.\n"
        "Run tools/setup_qnn_libs.sh <QNN_SDK_ROOT> first.")
endif()

add_library(rvc_qnn SHARED
    rvc_qnn_jni.cpp
    qnn_synth_runtime.cpp
)

# QNN's own headers reach for siblings with bare names (e.g.
# QnnSystemContext.h does `#include "QnnDevice.h"`), so the QNN/
# subdirectory has to be on the search path too.
target_include_directories(rvc_qnn PRIVATE
    ${QNN_INCLUDE_DIR}
    ${QNN_INCLUDE_DIR}/QNN
    ${QNN_INCLUDE_DIR}/QNN/HTP
    ${QNN_INCLUDE_DIR}/QNN/System
)

# log for __android_log_print, dl for dlopen/dlsym (used at runtime to
# resolve QnnInterface_getProviders et al. inside libQnnSystem.so).
target_link_libraries(rvc_qnn
    log
    dl
    android
)

target_compile_options(rvc_qnn PRIVATE
    -Wall
    -Wno-unused-parameter
    -fvisibility=hidden
)
