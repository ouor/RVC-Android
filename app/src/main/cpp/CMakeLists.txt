cmake_minimum_required(VERSION 3.18)
project(rvc_qnn_runners LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# QNN SDK headers are populated by tools/setup_qnn_libs.sh from a local
# install (see app/src/main/cpp/3rdparty/qnn/include/QNN). Functions
# from libQnnSystem.so / libQnnHtp.so are not linked statically — the
# runner binaries resolve them via dlopen at runtime, mirroring how
# QNN's own SampleApp links against the SDK.
set(QNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/qnn/include)

if(NOT EXISTS ${QNN_INCLUDE_DIR}/QNN/QnnInterface.h)
    message(FATAL_ERROR
        "QNN headers not found at ${QNN_INCLUDE_DIR}.\n"
        "Run tools/setup_qnn_libs.sh <QNN_SDK_ROOT> first.")
endif()

# Shared QNN host glue (init/setInput/execute/getOutput/destroy). Both
# the synth and HuBERT runner statically link it so each PIE binary is
# self-contained — no .so dependency between the two children.
add_library(qnn_runtime STATIC
    qnn_synth_runtime.cpp
)
target_include_directories(qnn_runtime PUBLIC
    ${QNN_INCLUDE_DIR}
    ${QNN_INCLUDE_DIR}/QNN
    ${QNN_INCLUDE_DIR}/QNN/HTP
    ${QNN_INCLUDE_DIR}/QNN/System
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(qnn_runtime PRIVATE
    -Wall
    -Wno-unused-parameter
    -fvisibility=hidden
)

# Common helper: build a PIE executable named lib<name>.so so the
# Android packager extracts it to nativeLibraryDir alongside the .so
# libraries. ProcessBuilder spawns it as a child; the SELinux domain
# transition is what unblocks unsigned PD on the DSP. See Phase κ for
# the original commit that introduced this trick.
function(add_qnn_runner name source)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES
        OUTPUT_NAME ${name}
        PREFIX "lib"
        SUFFIX ".so"
    )
    target_link_libraries(${name} PRIVATE qnn_runtime log dl android)
    target_compile_options(${name} PRIVATE
        -Wall
        -Wno-unused-parameter
        -fvisibility=hidden
    )
endfunction()

add_qnn_runner(rvc_synth_runner  rvc_synth_runner_main.cpp)
add_qnn_runner(rvc_hubert_runner rvc_hubert_runner_main.cpp)
