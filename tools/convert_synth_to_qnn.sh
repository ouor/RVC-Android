#!/usr/bin/env bash
# Compile a static-shape RVC synthesizer ONNX into a QNN context binary
# (.bin) targetable on Hexagon V79 (Snapdragon 8 Elite).
#
# Why context binaries
# --------------------
# QNN HTP runs pre-compiled context binaries far better than ONNX-via-ORT.
# Compilation happens off-device in three stages:
#
#   1. qnn-onnx-converter — ONNX → QNN IR (a .cpp source describing the
#      graph + a .bin file for weights). Pure model translation.
#   2. qnn-model-lib-generator — IR .cpp + weights → host-side aarch64
#      shared library (libmodel.so) the device can dlopen.
#   3. qnn-context-binary-generator — uses the QNN HTP backend to AOT-
#      compile the .so for a specific Hexagon arch. Output is a single
#      .bin context-binary the device loads in milliseconds with no
#      runtime graph compilation.
#
# Usage
# -----
#   bash tools/convert_synth_to_qnn.sh \
#     <QNN_SDK_ROOT> <INPUT_ONNX> [OUTPUT_DIR]
#
#   QNN_SDK_ROOT   e.g. /c/Users/.../2.45.0.260326
#   INPUT_ONNX     the static synth (T fixed) produced by
#                  tools/export_static_synthesizer.py
#   OUTPUT_DIR     defaults to ./qnn_build under the project root
#
# Env assumptions
# ---------------
# Run from a Python environment where torch + onnx are importable. The
# SDK's converter package is loaded via PYTHONPATH; we don't pip-install
# anything from the SDK into the env. Designed for the user's existing
# `default` conda env (Python 3.10).
set -e

if [ -z "${1-}" ] || [ -z "${2-}" ]; then
    echo "Usage: $0 <QNN_SDK_ROOT> <INPUT_ONNX> [OUTPUT_DIR]" >&2
    echo "Example:" >&2
    echo "  $0 /c/Users/hurwy/Downloads/_/2.45.0.260326 \\" >&2
    echo "     /c/Users/hurwy/Downloads/_/model_static_t192.onnx" >&2
    exit 1
fi

SDK="$1"
INPUT="$2"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${3:-$PROJECT/qnn_build}"

PY="${PY:-C:/Users/hurwy/miniconda3/envs/default/python.exe}"

if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: PY=$PY is not executable. Set PY=path/to/python or activate conda env." >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "error: input ONNX not found: $INPUT" >&2
    exit 1
fi
if [ ! -d "$SDK/lib/python" ]; then
    echo "error: $SDK does not look like a QNN SDK extraction (no lib/python)" >&2
    exit 1
fi

# Pick a host bin dir that exists on this machine.
if [ -d "$SDK/bin/x86_64-windows-msvc" ]; then
    HOST_BIN="$SDK/bin/x86_64-windows-msvc"
elif [ -d "$SDK/bin/x86_64-linux-clang" ]; then
    HOST_BIN="$SDK/bin/x86_64-linux-clang"
else
    echo "error: no host bin dir found in $SDK/bin" >&2
    exit 1
fi

# Windows Python parses PYTHONPATH with `;` separators and expects native
# `C:\…` paths; bash's `:` and `/c/…` conventions silently drop the entry
# (we hit `ModuleNotFoundError: qti` until we fixed this). cygpath -w gives
# us the right form regardless of which slashes the caller used.
if command -v cygpath >/dev/null 2>&1; then
    SDK_PY_PATH="$(cygpath -w "$SDK/lib/python")"
    PATH_SEP=";"
else
    SDK_PY_PATH="$SDK/lib/python"
    PATH_SEP=":"
fi
export PYTHONPATH="${SDK_PY_PATH}${PATH_SEP}${PYTHONPATH:-}"
export QNN_SDK_ROOT="$SDK"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

INPUT_BASE="$(basename "$INPUT" .onnx)"
IR_CPP="$OUTPUT_DIR/${INPUT_BASE}.cpp"
IR_BIN="$OUTPUT_DIR/${INPUT_BASE}.bin"
LIB_DIR="$OUTPUT_DIR/model_lib"
CTX_DIR="$OUTPUT_DIR/context_v79"

echo "==[1/3]== qnn-onnx-converter $INPUT → IR ($IR_CPP)"
"$PY" "$HOST_BIN/qnn-onnx-converter" \
    --input_network "$INPUT" \
    --output_path "$IR_CPP" \
    --keep_int64_inputs \
    --float_bitwidth 32

echo
echo "==[2/3]== qnn-model-lib-generator → aarch64-android libmodel.so"
"$PY" "$HOST_BIN/qnn-model-lib-generator" \
    -c "$IR_CPP" \
    -b "$IR_BIN" \
    -o "$LIB_DIR" \
    -t aarch64-android

LIB_SO="$LIB_DIR/aarch64-android/lib${INPUT_BASE}.so"
if [ ! -f "$LIB_SO" ]; then
    echo "error: expected $LIB_SO not produced" >&2
    ls -la "$LIB_DIR/aarch64-android" 2>&1 || true
    exit 1
fi

echo
echo "==[3/3]== qnn-context-binary-generator (HTP, V79) → ${INPUT_BASE}_v79.bin"
# The backend SO here is the HOST one that drives AOT compilation. The
# Hexagon arch (V79) is selected via dsp_arch in HTP backend config.
HTP_HOST_SO="$SDK/lib/x86_64-windows-msvc/QnnHtp.dll"
if [ ! -f "$HTP_HOST_SO" ]; then
    HTP_HOST_SO="$SDK/lib/x86_64-linux-clang/libQnnHtp.so"
fi
if [ ! -f "$HTP_HOST_SO" ]; then
    echo "error: no host QnnHtp backend found in $SDK/lib/x86_64-*" >&2
    exit 1
fi

# Minimal HTP backend extensions config: target Hexagon V79 (sm8750 = SE).
cat > "$OUTPUT_DIR/htp_v79_config.json" <<'JSON'
{
    "backend_extensions": {
        "shared_library_path": "",
        "config_file_path": ""
    },
    "graphs": [
        {
            "graph_names": ["*"],
            "fp16_relaxed_precision": 1,
            "vtcm_mb": 8,
            "O": 3,
            "hvx_threads": 4,
            "graph_priority": 0,
            "soc_model": 0,
            "dsp_arch": "v79"
        }
    ]
}
JSON

mkdir -p "$CTX_DIR"
"$HOST_BIN/qnn-context-binary-generator.exe" \
    --backend "$HTP_HOST_SO" \
    --model "$LIB_SO" \
    --binary_file "${INPUT_BASE}_v79" \
    --output_dir "$CTX_DIR" \
    --config_file "$OUTPUT_DIR/htp_v79_config.json" \
    || {
        echo
        echo "context-binary-generator failed. Re-run with --log_level VERBOSE if you need detail." >&2
        exit 1
    }

echo
echo "Done."
echo "  IR cpp:        $IR_CPP"
echo "  IR weights:    $IR_BIN"
echo "  host lib:      $LIB_SO"
echo "  context bin:   $CTX_DIR/${INPUT_BASE}_v79.bin"
echo
echo "Next: copy the .bin to /sdcard/Download/RVC/ on the device."
