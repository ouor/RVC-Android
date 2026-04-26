#!/usr/bin/env python3
"""Compile a ContentVec / HuBERT ONNX into a QNN context binary (.bin)
on Qualcomm AI Hub, targeting Samsung Galaxy S25 (Hexagon V79).

The ONNX shipped with voice-changer (02.onnx) has a dynamic time axis
on its `audio` input. AI Hub's compile job lets us pin that axis at
compile time via input_specs — we don't need to re-export the model
ourselves. We pin to 30720 samples (1.92 s @ 16 kHz) so each HuBERT
call covers exactly one synthesizer chunk and the pipeline keeps a
1:1 chunk structure.

We optionally extract a subgraph for a single output (`unit12` is what
v2 RVC checkpoints use); the original ContentVec graph also exposes
`units9` and `unit12s`, but compiling those branches just wastes DSP
time on activations we never read. The extraction uses
onnx.utils.extract_model, which preserves all transitive ops and
weights leading to the chosen output.

W8A16 quantization (Phase ω)
----------------------------
With --quantize w8a16 (and a calibration .npz produced by
tools/build_calibration_corpus.py), this script first runs an AI Hub
submit_quantize_job that turns the FP32 ONNX into a QDQ ONNX with
INT8 weights and INT16 activations, then compiles that QDQ ONNX into
a context binary. Two jobs ⇒ two dashboard URLs; both must succeed.

Usage
-----
    # FP16 (default):
    python tools/compile_hubert_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/content_vec_500.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --audio-samples 5120 \\
        --output-tensor unit12

    # W8A16:
    python tools/compile_hubert_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/content_vec_500.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --audio-samples 5120 \\
        --output-tensor unit12 \\
        --quantize w8a16 \\
        --calibration-npz qnn_build/calibration/audio_corpus.npz

Requires `pip install qai_hub onnx` and ~/.qai_hub/client.ini configured.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import qai_hub as hub


# Galaxy S25 (sm8750-ac, V79). Same target as the synth — the runtime
# loads the synth and HuBERT contexts on the same DSP, so they must
# agree on the architecture.
TARGET_DEVICE_NAME = "Samsung Galaxy S25"


def staticize_onnx_locally(src_path: Path, dst_path: Path,
                           input_dims: dict[str, tuple[int, ...]]) -> None:
    """Bake static input shapes into an ONNX before upload.

    AI Hub's submit_quantize_job rejects models with dynamic axes, and its
    compile-job route to staticize (--target_runtime onnx) sometimes mangles
    shape info on uncommon ops (we hit that with RMVPE's STFT). Doing the
    dim_param → dim_value substitution locally and rerunning shape inference
    avoids both problems.
    """
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(src_path))
    for inp in model.graph.input:
        if inp.name not in input_dims:
            continue
        new_dims = input_dims[inp.name]
        shape = inp.type.tensor_type.shape
        if len(shape.dim) != len(new_dims):
            sys.exit(f"input '{inp.name}' has rank {len(shape.dim)}, "
                     f"got {len(new_dims)} dims to set")
        for i, d in enumerate(new_dims):
            shape.dim[i].Clear()
            shape.dim[i].dim_value = int(d)

    # Strip any value_info entries — shape inference will rebuild them with
    # the new static shapes propagated through the graph.
    del model.graph.value_info[:]
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    onnx.save(inferred, str(dst_path))
    print(f"  static ONNX → {dst_path} "
          f"({dst_path.stat().st_size / (1024 * 1024):.1f} MiB)")


def extract_single_output(src_path: Path, output_name: str, dst_path: Path) -> None:
    """Trim the ONNX to keep only the subgraph that produces output_name.

    voice-changer's ContentVec export exposes three outputs (units9,
    unit12, unit12s). For our v2-only path we want exactly one. Compiling
    the trimmed graph keeps the .bin smaller and skips DSP work on
    branches we'd never read.
    """
    import onnx
    from onnx.utils import extract_model

    src_model = onnx.load(str(src_path))
    available_outputs = {o.name for o in src_model.graph.output}
    if output_name not in available_outputs:
        sys.exit(
            f"ONNX has no output '{output_name}' "
            f"(available: {sorted(available_outputs)})"
        )

    inputs = [i.name for i in src_model.graph.input]
    print(f"extract_model: inputs={inputs} outputs=[{output_name}]")
    extract_model(str(src_path), str(dst_path), inputs, [output_name])

    # extract_model leaves IO tensors duplicated in graph.value_info,
    # which AI Hub's onnx validator rejects ("Tensors {…} occur in
    # value_info but also in model IO"). Strip the offending entries.
    trimmed = onnx.load(str(dst_path))
    io_names = {i.name for i in trimmed.graph.input} | {o.name for o in trimmed.graph.output}
    keep = [vi for vi in trimmed.graph.value_info if vi.name not in io_names]
    if len(keep) != len(trimmed.graph.value_info):
        del trimmed.graph.value_info[:]
        trimmed.graph.value_info.extend(keep)
        onnx.save(trimmed, str(dst_path))
        print(f"  cleaned {len(io_names)} duplicate value_info entries")
    print(f"  -> {dst_path} ({dst_path.stat().st_size / (1024 * 1024):.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="path to ContentVec / HuBERT ONNX")
    parser.add_argument("--output-dir", default="./qnn_build/ai_hub",
                        help="local dir to download the compiled package into")
    parser.add_argument("--audio-samples", type=int, default=30720,
                        help="static audio length to pin (default 30720 = 1.92s @ 16kHz, "
                             "matches one synth chunk with WINDOW=160 × T=192)")
    parser.add_argument("--output-tensor", default="unit12",
                        help="ContentVec output to keep (unit12 for v2, units9 for v1, "
                             "unit12s for v2 with finalProj). Default: unit12.")
    parser.add_argument("--no-extract", action="store_true",
                        help="skip the onnx.utils.extract_model trim and upload "
                             "the multi-output ONNX as-is")
    parser.add_argument("--profile", action="store_true",
                        help="also submit a profile job for on-device latency stats")
    parser.add_argument("--htp-O", type=int, default=None, choices=(1, 2, 3),
                        help="HTP graph finalize optimization level (--qnn_options "
                             "default_graph_htp_optimizations=O=N). The default opts "
                             "try to put conv activations in VTCM; lower O can avoid "
                             "the 'requires N bytes of TCM' compile failure for big "
                             "graphs at the cost of slower runtime.")
    parser.add_argument("--model-id", default=None,
                        help="reuse a previously uploaded model_id instead of re-uploading. "
                             "Skips the extract+upload steps (handy for retrying with "
                             "different compile flags).")
    parser.add_argument("--quantize", choices=("w8a16",), default=None,
                        help="run an AI Hub submit_quantize_job before compile, "
                             "producing a QDQ ONNX with the chosen weight/act dtypes. "
                             "Currently only w8a16 (INT8 weights, INT16 activations).")
    parser.add_argument("--calibration-npz", default=None,
                        help="path to .npz from tools/build_calibration_corpus.py. "
                             "Required when --quantize is set. Picks the audio_5120 "
                             "or audio_30720 array based on --audio-samples.")
    parser.add_argument("--quantized-model-id", default=None,
                        help="reuse a previously produced quantized model_id; skips the "
                             "quantize step entirely. Mutually exclusive with --quantize.")
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
        if args.no_extract:
            upload_path = onnx_path
        else:
            trimmed = output_dir / f"{onnx_path.stem}_{args.output_tensor}_trim.onnx"
            extract_single_output(onnx_path, args.output_tensor, trimmed)
            upload_path = trimmed
        print(f"uploading {upload_path.name} ({upload_path.stat().st_size / (1024 * 1024):.1f} MiB)")
        raw = hub.upload_model(str(upload_path))
        print(f"  -> {raw.model_id}")

    # Pin the dynamic time axis. ContentVec's input is `audio[1, T]`
    # (fp32). Hexagon HTP needs concrete shapes at compile time.
    input_specs = {
        "audio": ((1, args.audio_samples), "float32"),
    }
    print(f"input_specs: {input_specs}")

    # ---- optional quantize step ------------------------------------------
    # AI Hub's submit_quantize_job rejects models with dynamic axes ("convert
    # dynamic shapes to static using a compile job that provides the input
    # shapes and targets ONNX runtime"). So when --quantize is set we run
    # an extra compile_job(--target_runtime onnx) first to bake static
    # shapes into a fresh ONNX, then feed that to the quantize job.
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
                     f"(available: {npz.files}). "
                     f"Re-run build_calibration_corpus.py with the right window.")
        windows = npz[key]  # (N, audio_samples) fp32
        calib_samples = [w.reshape(1, args.audio_samples).astype(np.float32) for w in windows]

        print("submitting pre-compile job → static ONNX (for quantize input)")
        static_job = hub.submit_compile_job(
            model=raw,
            device=device,
            input_specs=input_specs,
            options="--target_runtime onnx",
            name=f"rvc-hubert-static-{args.audio_samples}-{args.output_tensor}-staticize",
        )
        print(f"  job_id={static_job.job_id}, dashboard: {static_job.url}")
        print("waiting for staticize to finish…")
        static_model = static_job.get_target_model()
        if static_model is None:
            sys.exit("staticize compile failed — see dashboard URL above")
        print(f"static ONNX model: {static_model.model_id}")

        print(f"quantize: weights=INT8, activations=INT16, "
              f"calibration N={len(calib_samples)} from {npz_path}")
        quantize_job = hub.submit_quantize_job(
            model=static_model,
            calibration_data={"audio": calib_samples},
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"rvc-hubert-static-{args.audio_samples}-{args.output_tensor}-w8a16",
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
        name=f"rvc-hubert-static-{args.audio_samples}-{args.output_tensor}-{quantize_label}",
    )
    print(f"  job_id={compile_job.job_id}, dashboard: {compile_job.url}")
    print("waiting for compile to finish (typical 5-15 min)…")
    compiled = compile_job.get_target_model()
    if compiled is None:
        sys.exit("compile failed — see dashboard URL above for the build log")
    print(f"compiled model: {compiled.model_id}")

    suffix = f"_{quantize_label}" if quantize_label != "fp16" else ""
    archive = output_dir / f"{onnx_path.stem}_{args.output_tensor}_static_{args.audio_samples}{suffix}_qnn_v79.bin"
    print(f"downloading → {archive}")
    compiled.download(str(archive))
    print(f"  bytes: {archive.stat().st_size}")

    if args.profile:
        print("submitting profile job (on-device latency)")
        profile_job = hub.submit_profile_job(
            model=compiled,
            device=device,
            name=f"rvc-hubert-static-{args.audio_samples}-{args.output_tensor}-profile",
        )
        print(f"  job_id={profile_job.job_id}, dashboard: {profile_job.url}")
        results = profile_job.download_results(str(output_dir / "profile"))
        print(f"profile results: {results}")

    print()
    print("Next:")
    print(f"  1. push {archive} to /sdcard/Download/RVC/ on the device")
    print( "  2. point the app's HuBERT picker at the .bin (extension routing branches QNN vs ORT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
