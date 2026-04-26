cmake_minimum_required(VERSION 3.18)
project(rvc_synth_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# QNN SDK headers are populated by tools/setup_qnn_libs.sh from a local
# install (see app/src/main/cpp/3rdparty/qnn/include/QNN). Functions
# from libQnnSystem.so / libQnnHtp.so are not linked statically — the
# runner binary resolves them via dlopen at runtime, mirroring how
# QNN's own SampleApp links against the SDK.
set(QNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/qnn/include)

if(NOT EXISTS ${QNN_INCLUDE_DIR}/QNN/QnnInterface.h)
    message(FATAL_ERROR
        "QNN headers not found at ${QNN_INCLUDE_DIR}.\n"
        "Run tools/setup_qnn_libs.sh <QNN_SDK_ROOT> first.")
endif()

# add_executable, not add_library — see Phase κ commit message for why
# we ship this as a child process. PIE (-pie -fPIE) is the NDK default
# for executables on Android. The Android packager treats the output
# as a "library" for distribution purposes if its name starts with
# "lib" and ends in ".so", which is exactly what gets it extracted to
# nativeLibraryDir for ProcessBuilder to spawn — local-dream's trick.
add_executable(rvc_synth_runner
    rvc_synth_runner_main.cpp
    qnn_synth_runtime.cpp
)
set_target_properties(rvc_synth_runner PROPERTIES
    OUTPUT_NAME rvc_synth_runner
    PREFIX "lib"
    SUFFIX ".so"
)

target_include_directories(rvc_synth_runner PRIVATE
    ${QNN_INCLUDE_DIR}
    ${QNN_INCLUDE_DIR}/QNN
    ${QNN_INCLUDE_DIR}/QNN/HTP
    ${QNN_INCLUDE_DIR}/QNN/System
)

target_link_libraries(rvc_synth_runner
    log
    dl
    android
)

target_compile_options(rvc_synth_runner PRIVATE
    -Wall
    -Wno-unused-parameter
    -fvisibility=hidden
)
