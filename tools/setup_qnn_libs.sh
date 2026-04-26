#!/usr/bin/env bash
# Populate app/src/main/jniLibs and app/src/main/assets/qnn_skel with the
# host- and Hexagon-side libraries from a local QNN SDK install.
#
# We don't commit these binaries — too large (~112 MB total) and tied to a
# specific SDK version. Anyone building the app runs this script once
# pointing at their own SDK extraction.
#
# Why we need newer libs at all
# -----------------------------
# onnxruntime-android-qnn:1.22.0 ships QNN SDK 2.25 internally. That version
# predates Snapdragon 8 Elite (Hexagon V79) so QNN's CreateDevice rejects
# our config with QNN_DEVICE_ERROR_INVALID_CONFIG; HTP/GPU/DSP all return
# PLATFORM_NOT_SUPPORTED on the V79+Android 16 combination. SDK 2.45 has
# full V79 support, and Qualcomm maintains backward ABI compatibility on
# QnnInterface_t — so swapping out just the SOs (not ORT itself) lets the
# 1.22 binary call into the 2.45 driver.
#
# Usage
# -----
#   bash tools/setup_qnn_libs.sh /path/to/qnn-sdk-root
#
# Where qnn-sdk-root is the directory containing lib/aarch64-android/...
# (e.g. .../2.45.0.260326).
set -e

if [ -z "${1-}" ]; then
    echo "Usage: $0 <QNN_SDK_ROOT>" >&2
    echo "Example: $0 /c/Users/hurwy/Downloads/_/2.45.0.260326" >&2
    exit 1
fi

SDK="$1"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
JNI="$PROJECT/app/src/main/jniLibs/arm64-v8a"
ASSETS="$PROJECT/app/src/main/assets/qnn_skel"

if [ ! -d "$SDK/lib/aarch64-android" ] || [ ! -d "$SDK/lib/hexagon-v79/unsigned" ]; then
    echo "error: $SDK does not look like a QNN SDK extraction" >&2
    echo "  expected $SDK/lib/aarch64-android and $SDK/lib/hexagon-v79/unsigned" >&2
    exit 1
fi

mkdir -p "$JNI" "$ASSETS"

# Host-side aarch64 libs — go to jniLibs, packaged into APK as native libs.
# AGP packaging.jniLibs.pickFirsts in app/build.gradle.kts makes our copies
# win over the (older 2.25) ones bundled in onnxruntime-android-qnn.
for f in libQnnHtp.so libQnnSystem.so libQnnHtpV79Stub.so libQnnHtpPrepare.so; do
    src="$SDK/lib/aarch64-android/$f"
    [ -f "$src" ] || { echo "missing: $src" >&2; exit 1; }
    cp -v "$src" "$JNI/$f"
done

# Hexagon DSP-side libs — go to assets, extracted at runtime to a path
# accessible via ADSP_LIBRARY_PATH. They cannot live in jniLibs because
# Android would refuse to load them (Hexagon binary in an aarch64-v8a slot).
for f in libQnnHtpV79.so libQnnHtpV79Skel.so; do
    src="$SDK/lib/hexagon-v79/unsigned/$f"
    [ -f "$src" ] || { echo "missing: $src" >&2; exit 1; }
    cp -v "$src" "$ASSETS/$f"
done

echo
echo "Done. APK will pack QNN 2.45 libs at:"
echo "  $JNI"
echo "  $ASSETS"
