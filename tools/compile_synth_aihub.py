#!/usr/bin/env python3
"""Compile a static-shape RVC synthesizer ONNX into a QNN context binary
(.bin) on Qualcomm AI Hub, targeting Samsung Galaxy S25 (Hexagon V79).

Why AI Hub instead of the local SDK
-----------------------------------
The local QNN SDK 2.45 qnn-onnx-converter chokes on the synth's pattern
`g = self.emb_g(sid).unsqueeze(-1)` — it mis-parses the Unsqueeze axes
input and emits garbage shape values, killing translation before any op
gets through. AI Hub's pipeline handles those edge cases plus the
attention/Concat shape patterns the local converter also warned about,
and it runs the actual on-device profile so we know FP16 dispatch onto
Hexagon V79 worked before we ship the binary.

Usage
-----
    python tools/compile_synth_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/model_static_t192.onnx \\
        --output-dir ./qnn_build/ai_hub

Requires `pip install qai_hub` and ~/.qai_hub/client.ini configured.
"""
import argparse
import sys
from pathlib import Path

import qai_hub as hub


# Galaxy S25 vanilla (sm8750-ac, V79, soc-model 69). The "(Family)" alias
# also exists; we pin to the specific model for reproducibility.
TARGET_DEVICE_NAME = "Samsung Galaxy S25"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path to static-shape ONNX")
    parser.add_argument("--output-dir", default="./qnn_build/ai_hub",
                        help="local dir to download the compiled package into")
    parser.add_argument("--frames", type=int, default=192,
                        help="static T baked into the ONNX (default 192)")
    parser.add_argument("--channels", type=int, default=768,
                        help="emb_channels (default 768 for v2 RVC)")
    parser.add_argument("--inter-channels", type=int, default=192,
                        help="synth's inter_channels = rand_noise channel dim (default 192)")
    parser.add_argument("--external-noise", action="store_true", default=True,
                        help="model expects rand_noise input (the QNN-friendly variant)")
    parser.add_argument("--profile", action="store_true",
                        help="also submit a profile job for on-device latency stats")
    args = parser.parse_args()

    onnx_path = Path(args.input)
    if not onnx_path.is_file():
        sys.exit(f"input ONNX not found: {onnx_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pin to the exact device whose chipset matches our test phone. AI Hub
    # uses this to choose the correct HTP backend, soc_model, and DSP arch
    # at compile time — we don't have to set them ourselves.
    device = hub.Device(name=TARGET_DEVICE_NAME)
    print(f"target device: {device.name} (attrs: {device.attributes})")

    print(f"uploading {onnx_path.name} ({onnx_path.stat().st_size / (1024 * 1024):.1f} MiB)")
    raw = hub.upload_model(str(onnx_path))
    print(f"  -> {raw.model_id}")

    # The compile job needs explicit input shapes/dtypes when the ONNX
    # carries any axis the runtime might re-interpret. Our static export
    # has them all baked in, but passing input_specs is the safe form.
    T = args.frames
    C = args.channels
    IC = args.inter_channels
    input_specs = {
        "feats":  ((1, T, C), "float16"),
        "p_len":  ((1,),       "int64"),
        "pitch":  ((1, T),     "int64"),
        "pitchf": ((1, T),     "float32"),
        "sid":    ((1,),       "int64"),
    }
    if args.external_noise:
        # Matches the patched forward() in tools/export_static_synthesizer.py:
        # noise replaces torch.randn_like(m_p), so shape is m_p's shape =
        # (1, inter_channels, T). Hexagon HTP cannot generate random
        # numbers, hence the externalisation.
        input_specs["rand_noise"] = ((1, IC, T), "float16")
    print(f"input_specs: {input_specs}")

    print("submitting compile job → QNN context binary")
    compile_job = hub.submit_compile_job(
        model=raw,
        device=device,
        input_specs=input_specs,
        # --target_runtime qnn_context_binary picks .bin context-binary
        #   as the artifact (vs ONNX-bundled QNN, etc.).
        # --truncate_64bit_io is required because the synth's p_len/pitch/
        #   sid inputs are int64 — Hexagon HTP doesn't accept int64 IO and
        #   AI Hub fails the compile otherwise. Truncation to int32 is
        #   safe: the values are frame counts, mel bin indices in [1,255],
        #   and a small speaker id, none of which exceed int32 range.
        # --quantize_full_type w8a16 (not used) would buy more speedup but
        #   needs calibration data. We stay FP16 to match the synth export.
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
        name=f"rvc-synth-static-t{T}",
    )
    print(f"  job_id={compile_job.job_id}, dashboard: {compile_job.url}")
    print("waiting for compile to finish (typical 5-15 min)…")
    compiled = compile_job.get_target_model()
    if compiled is None:
        sys.exit("compile failed — see dashboard URL above for the build log")
    print(f"compiled model: {compiled.model_id}")

    archive = output_dir / f"{onnx_path.stem}_qnn_v79.bin"
    print(f"downloading → {archive}")
    compiled.download(str(archive))
    print(f"  bytes: {archive.stat().st_size}")

    if args.profile:
        print("submitting profile job (on-device latency)")
        profile_job = hub.submit_profile_job(
            model=compiled,
            device=device,
            name=f"rvc-synth-static-t{T}-profile",
        )
        print(f"  job_id={profile_job.job_id}, dashboard: {profile_job.url}")
        results = profile_job.download_results(str(output_dir / "profile"))
        print(f"profile results: {results}")

    print()
    print("Next:")
    print(f"  1. push {archive} to /sdcard/Download/RVC/ on the device")
    print( "  2. point the app's synthesizer picker at the .bin")
    print( "  3. native QNN runtime (Phase η) loads it via QnnContext_createFromBinary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
