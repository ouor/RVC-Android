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

W8A16 quantization (Phase ω)
----------------------------
With --quantize w8a16 (and a calibration .npz produced by
tools/build_calibration_corpus.py), this script first runs an AI Hub
submit_quantize_job that turns the FP32 ONNX into a QDQ ONNX with
INT8 weights and INT16 activations, then compiles that QDQ ONNX into
a context binary.

The threshold input is a runtime scalar — for calibration we feed
the same fixed value (0.3) used by the on-device pipeline, so its
quantization scale collapses to a tight range.

Usage
-----
    # FP16 (default):
    python tools/compile_rmvpe_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/MMVCServerSIO/pretrain/rmvpe.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --audio-samples 30720

    # W8A16:
    python tools/compile_rmvpe_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/MMVCServerSIO/pretrain/rmvpe.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --audio-samples 30720 \\
        --quantize w8a16 \\
        --calibration-npz qnn_build/calibration/audio_corpus.npz

Requires `pip install qai_hub onnx` and ~/.qai_hub/client.ini configured.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import qai_hub as hub


TARGET_DEVICE_NAME = "Samsung Galaxy S25"


def staticize_onnx_locally(src_path: Path, dst_path: Path,
                           input_dims: dict[str, tuple[int, ...]]) -> None:
    """Bake static input shapes into an ONNX before upload.

    Plain onnx.shape_inference is not enough for RMVPE — its STFT op
    leaves the next Reshape's output as `[1, unk__4]`, which AI Hub's
    quantize-time ORT loader rejects with
    "STFT window input must have rank = 1". onnxsim does aggressive
    constant folding and propagates the static signal shape through
    STFT, so the loader stops complaining.

    AI Hub's compile-job route to staticize (--target_runtime onnx)
    has the same shape-mangling problem, so doing it locally is also
    the only path.
    """
    import onnx
    import onnxsim

    src_model = onnx.load(str(src_path))
    overwrite = {name: list(dims) for name, dims in input_dims.items()}
    simplified, ok = onnxsim.simplify(src_model, overwrite_input_shapes=overwrite)
    if not ok:
        sys.exit("onnxsim.simplify reported failure — check the model")
    onnx.save(simplified, str(dst_path))
    print(f"  static ONNX → {dst_path} "
          f"({dst_path.stat().st_size / (1024 * 1024):.1f} MiB)")


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
    parser.add_argument("--quantize", choices=("w8a16",), default=None,
                        help="run an AI Hub submit_quantize_job before compile, "
                             "producing a QDQ ONNX with INT8 weights / INT16 activations.")
    parser.add_argument("--calibration-npz", default=None,
                        help="path to .npz from tools/build_calibration_corpus.py. "
                             "Required when --quantize is set.")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="voicing threshold used for both calibration and runtime "
                             "(default 0.3 — matches the on-device pipeline).")
    parser.add_argument("--quantized-model-id", default=None,
                        help="reuse a previously produced quantized model_id; mutually "
                             "exclusive with --quantize.")
    args = parser.parse_args()

    if args.quantize and args.quantized_model_id:
        sys.exit("--quantize and --quantized-model-id are mutually exclusive")
    if args.quantize and not args.calibration_npz:
        sys.exit("--quantize requires --calibration-npz")

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
        # Locally bake static shapes into the ONNX before upload. AI Hub's
        # compile-to-static-onnx step corrupts the STFT window input on this
        # graph; doing it ourselves with onnx.shape_inference avoids that
        # path entirely. Both quantize and the final compile_job accept
        # already-static models without needing the staticize round trip.
        static_local = output_dir / f"{onnx_path.stem}_static_{args.audio_samples}.onnx"
        print(f"locally staticizing {onnx_path.name} → "
              f"waveform[1, {args.audio_samples}], threshold[1]")
        staticize_onnx_locally(onnx_path, static_local, {
            "waveform": (1, args.audio_samples),
            "threshold": (1,),
        })
        print(f"uploading {static_local.name} "
              f"({static_local.stat().st_size / (1024 * 1024):.1f} MiB)")
        raw = hub.upload_model(str(static_local))
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

    # ---- optional quantize step ------------------------------------------
    # submit_quantize_job needs static shapes; we baked them in locally
    # before upload above. AI Hub's compile-to-static-onnx path mangles
    # RMVPE's STFT window input, so we sidestep it entirely.
    quantize_label = "fp16"
    model_for_compile = raw
    if args.quantize:
        npz_path = Path(args.calibration_npz)
        if not npz_path.is_file():
            sys.exit(f"calibration npz not found: {npz_path}")
        npz = np.load(str(npz_path))
        key = f"audio_{args.audio_samples}"
        if key not in npz.files:
            sys.exit(f"calibration npz {npz_path.name} has no array '{key}' "
                     f"(available: {npz.files})")
        windows = npz[key]
        wave_samples = [w.reshape(1, args.audio_samples).astype(np.float32) for w in windows]
        # Threshold is a fixed scalar in the on-device pipeline. Feeding the
        # same value for every calibration sample collapses its activation
        # range so the quantizer doesn't waste a scale on noise.
        thr_val = np.array([args.threshold], dtype=np.float32)
        thr_samples = [thr_val.copy() for _ in wave_samples]

        # Already-static (locally staticized at upload time) — feed straight
        # into quantize without a round-trip AI Hub compile_to_onnx.
        print(f"quantize: weights=INT8, activations=INT16, "
              f"calibration N={len(wave_samples)} from {npz_path}")
        quantize_job = hub.submit_quantize_job(
            model=raw,
            calibration_data={
                "waveform": wave_samples,
                "threshold": thr_samples,
            },
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"rvc-rmvpe-static-{args.audio_samples}-w8a16",
        )
        print(f"  job_id={quantize_job.job_id}, dashboard: {quantize_job.url}")
        print("waiting for quantize to finish (typical 5-15 min)…")
        model_for_compile = quantize_job.get_target_model()
        if model_for_compile is None:
            sys.exit("quantize failed — see dashboard URL above")
        print(f"quantized QDQ model: {model_for_compile.model_id}")
        quantize_label = "w8a16"
    elif args.quantized_model_id:
        print(f"reusing quantized model_id: {args.quantized_model_id}")
        model_for_compile = hub.get_model(args.quantized_model_id)
        quantize_label = "w8a16"

    options_parts = ["--target_runtime qnn_context_binary", "--truncate_64bit_io"]
    if args.htp_O is not None:
        options_parts.append(
            f"--qnn_options default_graph_htp_optimizations=O={args.htp_O}",
        )
    options = " ".join(options_parts)
    print(f"compile options: {options}")

    print("submitting compile job → QNN context binary")
    compile_job = hub.submit_compile_job(
        model=model_for_compile,
        device=device,
        input_specs=input_specs,
        options=options,
        name=f"rvc-rmvpe-static-{args.audio_samples}-{quantize_label}",
    )
    print(f"  job_id={compile_job.job_id}, dashboard: {compile_job.url}")
    print("waiting for compile to finish (typical 5-15 min)…")
    compiled = compile_job.get_target_model()
    if compiled is None:
        sys.exit("compile failed — see dashboard URL above for the build log")
    print(f"compiled model: {compiled.model_id}")

    suffix = f"_{quantize_label}" if quantize_label != "fp16" else ""
    archive = output_dir / f"{onnx_path.stem}_static_{args.audio_samples}{suffix}_qnn_v79.bin"
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
