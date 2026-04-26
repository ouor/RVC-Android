#!/usr/bin/env bash
# Recursively pull vendor libs needed (transitively) by libcdsprpc.so so
# the rvc_synth_runner can dlopen them by basename. Android's app linker
# namespace permits /data/app/.../lib/arm64 (where jniLibs end up) but
# not /vendor/lib64 — so we have to ship every vendor dep ourselves.
#
# Usage:
#   bash tools/pull_vendor_deps.sh [seed_lib_basename ...]
#
# Defaults to a seed that matches what fails today: libcdsprpc.so. The
# script reads each lib's NEEDED entries, locates anything that isn't
# already in /system or /apex on the device, pulls it from /vendor/lib64
# into app/src/main/jniLibs/arm64-v8a/, and recurses until the set of
# missing deps is empty.
set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
JNI="$PROJECT/app/src/main/jniLibs/arm64-v8a"
NDK="${NDK:-C:/Users/hurwy/AppData/Local/Android/Sdk/ndk/27.1.12297006}"
READELF="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-readelf.exe"
ADB="${ADB:-C:/Users/hurwy/AppData/Local/Android/Sdk/platform-tools/adb.exe}"

if [ ! -x "$READELF" ]; then
    echo "error: llvm-readelf not at $READELF" >&2
    exit 1
fi
mkdir -p "$JNI"

SEED=("${@:-libcdsprpc.so}")
queue=("${SEED[@]}")
processed=()

# Treat anything already discoverable in /system/{lib64,lib64/bootstrap},
# /system_ext/lib64, or /apex/com.android.runtime/lib64/bionic as "free"
# — Android's default namespace already searches those.
SYSTEM_HAS=$(MSYS_NO_PATHCONV=1 "$ADB" shell '
  for d in /system/lib64 /system/lib64/bootstrap /system_ext/lib64 /apex/com.android.runtime/lib64/bionic; do
    ls "$d" 2>/dev/null
  done
' | tr '\r' '\n' | sort -u)

was_processed() {
    local needle="$1"
    for p in "${processed[@]}"; do
        [ "$p" = "$needle" ] && return 0
    done
    return 1
}

while [ ${#queue[@]} -gt 0 ]; do
    lib="${queue[0]}"
    queue=("${queue[@]:1}")

    if was_processed "$lib"; then continue; fi
    processed+=("$lib")

    # System libs need no bundling.
    if echo "$SYSTEM_HAS" | grep -qx "$lib"; then
        echo "skip system: $lib"
        continue
    fi

    target="$JNI/$lib"
    if [ ! -f "$target" ]; then
        echo "pull /vendor/lib64/$lib → $target"
        export MSYS_NO_PATHCONV=1
        if ! "$ADB" pull "/vendor/lib64/$lib" "$target" >/dev/null 2>&1; then
            echo "  warn: not found in /vendor/lib64/$lib (may be fine if optional)" >&2
            continue
        fi
    fi

    # Walk this lib's NEEDED entries and queue anything unknown.
    while IFS= read -r dep; do
        [ -z "$dep" ] && continue
        if was_processed "$dep"; then continue; fi
        # Already bundled?
        if [ -f "$JNI/$dep" ]; then
            processed+=("$dep")
            continue
        fi
        queue+=("$dep")
    done < <("$READELF" -d "$target" 2>/dev/null \
        | awk '/\(NEEDED\)/ { match($0, /\[([^]]+)\]/, a); print a[1] }')
done

echo
echo "Bundled libs in $JNI:"
ls "$JNI"
