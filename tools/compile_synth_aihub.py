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

W8A16 quantization (Phase ω)
----------------------------
With --quantize w8a16 the script first runs an AI Hub
submit_quantize_job (the synth ONNX is already static, so no staticize
step needed) and then compiles the QDQ output to a context binary.
Calibration data must match the synth's six inputs; produce it with
tools/build_synth_calibration.py — that runs HuBERT and RMVPE on the
audio corpus from build_calibration_corpus.py to get realistic
feats/pitch/pitchf, plus seeded Gaussian rand_noise.

Usage
-----
    # FP16 (default):
    python tools/compile_synth_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/model_static_t192_qnn.onnx \\
        --output-dir ./qnn_build/ai_hub

    # W8A16:
    python tools/compile_synth_aihub.py \\
        --input C:/Users/hurwy/Downloads/_/model_static_t192_qnn.onnx \\
        --output-dir ./qnn_build/ai_hub \\
        --quantize w8a16 \\
        --calibration-npz qnn_build/calibration/synth_corpus.npz

Requires `pip install qai_hub` and ~/.qai_hub/client.ini configured.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import qai_hub as hub


def zero_out_random_ops(src_path: Path, dst_path: Path) -> int:
    """Replace any RandomUniformLike node with a Constant of zeros.

    NSF SineGen has a torch.rand_like seeding the initial phase offset.
    For the FP16 path it survives onto the DSP, but once the input is
    quantized (UFIXED) the HTP backend rejects the op:
      "Unsupported input/output datatypes requested for the HTP Op
       'RandomUniformLike'".
    Replacing the random output with zeros keeps the graph runnable and
    matches what tools/export_static_synthesizer.py already does to
    torch.randn{,_like} via the --external-noise patch.
    """
    import onnx
    model = onnx.load(str(src_path))
    patched = 0
    for i, n in enumerate(list(model.graph.node)):
        if n.op_type != "RandomUniformLike":
            continue
        # Find the output shape from value_info so the replacement matches.
        out_name = n.output[0]
        shape: tuple[int, ...] = (1, 1)  # safe default
        for vi in list(model.graph.value_info) + list(model.graph.output):
            if vi.name == out_name:
                shape = tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
                break
        zero = onnx.numpy_helper.from_array(
            np.zeros(shape, dtype=np.float32), name=out_name + "_zero",
        )
        new_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=[out_name],
            name=n.name + "_zero", value=zero,
        )
        model.graph.node.remove(n)
        model.graph.node.insert(i, new_node)
        patched += 1
        print(f"  replaced {n.name} → Constant zeros{list(shape)}")
    if patched:
        onnx.save(model, str(dst_path))
    return patched


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
    parser.add_argument("--model-id", default=None,
                        help="reuse a previously uploaded model_id instead of re-uploading.")
    parser.add_argument("--quantize", choices=("w8a16",), default=None,
                        help="run AI Hub submit_quantize_job before compile, "
                             "producing a QDQ ONNX with INT8 weights / INT16 activations.")
    parser.add_argument("--calibration-npz", default=None,
                        help="path to .npz from tools/build_synth_calibration.py "
                             "(contains feats/p_len/pitch/pitchf/sid/rand_noise).")
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

    # Pin to the exact device whose chipset matches our test phone. AI Hub
    # uses this to choose the correct HTP backend, soc_model, and DSP arch
    # at compile time — we don't have to set them ourselves.
    device = hub.Device(name=TARGET_DEVICE_NAME)
    print(f"target device: {device.name} (attrs: {device.attributes})")

    if args.model_id:
        print(f"reusing uploaded model_id: {args.model_id}")
        raw = hub.get_model(args.model_id)
    else:
        upload_path = onnx_path
        # When quantizing, scrub any RandomUniformLike op first — the HTP
        # backend doesn't support its quantized I/O signature.
        if args.quantize:
            patched_path = output_dir / f"{onnx_path.stem}_no_random.onnx"
            n = zero_out_random_ops(onnx_path, patched_path)
            if n:
                upload_path = patched_path
                print(f"  zeroed {n} RandomUniformLike op(s) → {patched_path.name}")
        print(f"uploading {upload_path.name} ({upload_path.stat().st_size / (1024 * 1024):.1f} MiB)")
        raw = hub.upload_model(str(upload_path))
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

    # ---- optional quantize step ------------------------------------------
    quantize_label = "fp16"
    model_for_compile = raw
    if args.quantize:
        npz_path = Path(args.calibration_npz)
        if not npz_path.is_file():
            sys.exit(f"calibration npz not found: {npz_path}")
        npz = np.load(str(npz_path))
        # AI Hub validates calibration_data keys against the ONNX input
        # order, so the dict must be inserted in that order. The synth
        # ONNX schema is [feats, p_len, pitch, pitchf, sid, (rand_noise)].
        required = ["feats", "p_len", "pitch", "pitchf", "sid"]
        if args.external_noise:
            required.append("rand_noise")
        missing = [n for n in required if n not in npz.files]
        if missing:
            sys.exit(f"calibration npz {npz_path.name} missing arrays: {missing}")

        # Each array is shaped (N, *input_shape). AI Hub wants per-input
        # lists of single-batch tensors.
        n_samples = npz["feats"].shape[0]
        print(f"quantize: weights=INT8, activations=INT16, "
              f"calibration N={n_samples} from {npz_path}")
        calibration_data: dict[str, list[np.ndarray]] = {}
        for name in required:
            arr = npz[name]
            if arr.shape[0] != n_samples:
                sys.exit(f"calibration array '{name}' has N={arr.shape[0]} "
                         f"but feats has N={n_samples}")
            calibration_data[name] = [arr[i] for i in range(n_samples)]

        quantize_job = hub.submit_quantize_job(
            model=raw,
            calibration_data=calibration_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"rvc-synth-static-t{T}-w8a16",
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

    print("submitting compile job → QNN context binary")
    compile_job = hub.submit_compile_job(
        model=model_for_compile,
        device=device,
        input_specs=input_specs,
        # --target_runtime qnn_context_binary picks .bin context-binary
        #   as the artifact (vs ONNX-bundled QNN, etc.).
        # --truncate_64bit_io is required because the synth's p_len/pitch/
        #   sid inputs are int64 — Hexagon HTP doesn't accept int64 IO and
        #   AI Hub fails the compile otherwise. Truncation to int32 is
        #   safe: the values are frame counts, mel bin indices in [1,255],
        #   and a small speaker id, none of which exceed int32 range.
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
        name=f"rvc-synth-static-t{T}-{quantize_label}",
    )
    print(f"  job_id={compile_job.job_id}, dashboard: {compile_job.url}")
    print("waiting for compile to finish (typical 5-15 min)…")
    compiled = compile_job.get_target_model()
    if compiled is None:
        sys.exit("compile failed — see dashboard URL above for the build log")
    print(f"compiled model: {compiled.model_id}")

    suffix = f"_{quantize_label}" if quantize_label != "fp16" else ""
    archive = output_dir / f"{onnx_path.stem}{suffix}_qnn_v79.bin"
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
