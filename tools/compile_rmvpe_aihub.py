#!/usr/bin/env python3
"""Compile an RMVPE pitch-extraction ONNX into a QNN context binary
(.bin) on Qualcomm AI Hub, targeting Samsung Galaxy S25 (Hexagon V79).

We pin the audio dimension to a fixed length per call. The natural
choice — like HuBERT — is one synth chunk (30720 samples = 1.92 s
@ 16 kHz). RMVPE outputs at 100 Hz with hop=160, so the pitch tensor
is exactly 192 frames per chunk, lining up 1:1 with the synth's
static T axis.

Boundary trade-off: RMVPE has a CNN+GRU stack and benefits from
seeing the whole clip. Per-chunk loses GRU context across boundaries
— pitch may have small jumps at chunk seams. For PoC quality this
appears tolerable; revisit if audible.

Usage
-----
    python tools/compile_rmvpe_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/03.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --audio-samples 30720

Requires `pip install qai_hub onnx` and ~/.qai_hub/client.ini configured.
"""
import argparse
import sys
from pathlib import Path

import qai_hub as hub


TARGET_DEVICE_NAME = "Samsung Galaxy S25"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="path to RMVPE ONNX")
    parser.add_argument("--output-dir", default="./qnn_build/ai_hub",
                        help="local dir to download the compiled package into")
    parser.add_argument("--audio-samples", type=int, default=30720,
                        help="static audio length to pin (default 30720 = 1.92s @ 16kHz, "
                             "= one synth chunk → exactly 192 pitch frames @ 100 Hz)")
    parser.add_argument("--profile", action="store_true",
                        help="also submit a profile job for on-device latency stats")
    parser.add_argument("--htp-O", type=int, default=None, choices=(1, 2, 3),
                        help="HTP graph finalize optimization level. Lower can avoid "
                             "TCM-overflow compile failures at the cost of slower runtime.")
    parser.add_argument("--model-id", default=None,
                        help="reuse a previously uploaded model_id instead of re-uploading.")
    args = parser.parse_args()

    onnx_path = Path(args.input)
    if not onnx_path.is_file():
        sys.exit(f"input ONNX not found: {onnx_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = hub.Device(name=TARGET_DEVICE_NAME)
    print(f"target device: {device.name} (attrs: {device.attributes})")

    if args.model_id:
        print(f"reusing uploaded model_id: {args.model_id}")
        raw = hub.get_model(args.model_id)
    else:
        print(f"uploading {onnx_path.name} ({onnx_path.stat().st_size / (1024 * 1024):.1f} MiB)")
        raw = hub.upload_model(str(onnx_path))
        print(f"  -> {raw.model_id}")

    # voice-changer's RMVPE export takes (waveform, threshold). We pin
    # both. Threshold is a fixed scalar but AI Hub still wants the
    # dtype/shape in input_specs. 0.03 is RMVPE's typical voicing
    # threshold; the runner sends the same value at runtime.
    input_specs = {
        "waveform": ((1, args.audio_samples), "float32"),
        "threshold": ((1,), "float32"),
    }
    print(f"input_specs: {input_specs}")

    options_parts = ["--target_runtime qnn_context_binary", "--truncate_64bit_io"]
    if args.htp_O is not None:
        options_parts.append(
            f"--qnn_options default_graph_htp_optimizations=O={args.htp_O}",
        )
    options = " ".join(options_parts)
    print(f"compile options: {options}")

    print("submitting compile job → QNN context binary")
    compile_job = hub.submit_compile_job(
        model=raw,
        device=device,
        input_specs=input_specs,
        options=options,
        name=f"rvc-rmvpe-static-{args.audio_samples}",
    )
    print(f"  job_id={compile_job.job_id}, dashboard: {compile_job.url}")
    print("waiting for compile to finish (typical 5-15 min)…")
    compiled = compile_job.get_target_model()
    if compiled is None:
        sys.exit("compile failed — see dashboard URL above for the build log")
    print(f"compiled model: {compiled.model_id}")

    archive = output_dir / f"{onnx_path.stem}_static_{args.audio_samples}_qnn_v79.bin"
    print(f"downloading → {archive}")
    compiled.download(str(archive))
    print(f"  bytes: {archive.stat().st_size}")

    if args.profile:
        print("submitting profile job (on-device latency)")
        profile_job = hub.submit_profile_job(
            model=compiled,
            device=device,
            name=f"rvc-rmvpe-static-{args.audio_samples}-profile",
        )
        print(f"  job_id={profile_job.job_id}, dashboard: {profile_job.url}")
        results = profile_job.download_results(str(output_dir / "profile"))
        print(f"profile results: {results}")

    print()
    print("Next:")
    print(f"  1. push {archive} to /sdcard/Download/RVC/ on the device")
    print( "  2. point the app's RMVPE picker at the .bin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
