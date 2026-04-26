#!/usr/bin/env python3
"""Re-export an RVC synthesizer .pth checkpoint to ONNX with a fixed T axis.

Patterned on voice-changer's
  server/voice_changer/RVC/onnxExporter/export2onnx.py
but with `dynamic_axes` removed. NNAPI/QNN partitioners refuse most ops on
dynamic shapes — this is what kept our previous export from getting NPU
dispatch. Pinning T to a single value lets the EP plan the entire graph.

Only the SYNTHESIZER is exported here; HuBERT/ContentVec and RMVPE come from
external community-distributed ONNX files we do not control. The synth is
~75% of inference time, so static-shape on it is the highest-impact step.

Usage
-----
  python tools/export_static_synthesizer.py \\
    --checkpoint /path/to/model.pth \\
    --voice-changer /path/to/voice-changer \\
    --output ./model_static_t192.onnx \\
    --frames 192 \\
    [--half] \\
    [--no-simplify]

Pick --frames to match the chunk length your runtime will feed at inference
time. 192 is voice-changer's realtime convention (1.92s of synth at the
100fps frame rate after HuBERT 2x upsampling). The exported model will only
accept feats/pitch/pitchf with exactly this T.

Dependencies (install in the same venv as voice-changer)
  pip install torch onnx onnxsim
plus voice-changer's own deps (fairseq, etc.) since we import its
SynthesizerTrn* classes.
"""
import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="path to RVC .pth")
    parser.add_argument("--voice-changer", required=True, help="path to voice-changer repo root")
    parser.add_argument("--output", required=True, help="output .onnx path")
    parser.add_argument(
        "--frames",
        type=int,
        default=192,
        help="fixed T (default 192 = 1.92s of synth at 100fps post-2x-upsample)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="export in float16 (matches voice-changer's is_half=True default; needs CUDA)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="skip onnx-simplifier (it can fail on some graphs; raw export still works)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    args = parser.parse_args()

    server_dir = os.path.join(args.voice_changer, "server")
    if not os.path.isdir(server_dir):
        sys.exit(f"voice-changer server dir not found: {server_dir}")
    sys.path.insert(0, server_dir)

    import torch
    from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_ONNX import (
        SynthesizerTrnMs256NSFsid_ONNX,
    )
    from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_nono_ONNX import (
        SynthesizerTrnMs256NSFsid_nono_ONNX,
    )
    from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (
        SynthesizerTrnMs768NSFsid_ONNX,
    )
    from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_nono_ONNX import (
        SynthesizerTrnMs768NSFsid_nono_ONNX,
    )

    cpt = torch.load(args.checkpoint, map_location="cpu")
    if "config" not in cpt or "weight" not in cpt:
        sys.exit("checkpoint missing 'config'/'weight' keys; not an RVC model")

    sr = cpt["config"][-1]
    f0 = bool(cpt.get("f0", 1))
    version = cpt.get("version", "v1")
    use_v2 = version == "v2"
    emb_channels = 768 if use_v2 else 256
    emb_layer = 12 if use_v2 else 9
    use_final_proj = not use_v2

    print(
        f"loaded checkpoint: sr={sr}, f0={f0}, version={version}, "
        f"emb_channels={emb_channels}, half={args.half}"
    )

    is_half = args.half
    if is_half and not torch.cuda.is_available():
        sys.exit(
            "--half requires CUDA. Either install a CUDA torch build, or drop --half "
            "to export FP32 (works on CPU, ~2x larger ONNX)."
        )
    device = torch.device("cuda") if is_half else torch.device("cpu")

    if use_v2 and f0:
        net_g = SynthesizerTrnMs768NSFsid_ONNX(*cpt["config"], is_half=is_half)
    elif use_v2 and not f0:
        net_g = SynthesizerTrnMs768NSFsid_nono_ONNX(*cpt["config"])
    elif (not use_v2) and f0:
        net_g = SynthesizerTrnMs256NSFsid_ONNX(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono_ONNX(*cpt["config"])

    net_g.eval().to(device)
    net_g.load_state_dict(cpt["weight"], strict=False)
    if is_half:
        net_g = net_g.half()

    T = args.frames
    feats_dtype = torch.float16 if is_half else torch.float32
    feats = torch.zeros(1, T, emb_channels, dtype=feats_dtype, device=device)
    p_len = torch.tensor([T], dtype=torch.int64, device=device)
    sid = torch.tensor([0], dtype=torch.int64, device=device)

    if f0:
        pitch = torch.zeros(1, T, dtype=torch.int64, device=device)
        pitchf = torch.zeros(1, T, dtype=torch.float32, device=device)
        input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]
        inputs = (feats, p_len, pitch, pitchf, sid)
    else:
        input_names = ["feats", "p_len", "sid"]
        inputs = (feats, p_len, sid)

    output_names = ["audio"]

    print(f"exporting fixed T={T} → {args.output}")
    # No dynamic_axes — every shape is concrete. This is the whole point.
    torch.onnx.export(
        net_g,
        inputs,
        args.output,
        do_constant_folding=False,
        opset_version=args.opset,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    import onnx

    model = onnx.load(args.output)

    if not args.no_simplify:
        try:
            from onnxsim import simplify

            simplified, ok = simplify(model)
            if ok:
                model = simplified
                print("onnx-simplifier: ok")
            else:
                print("onnx-simplifier: validation failed; keeping raw export", file=sys.stderr)
        except ImportError:
            print(
                "onnx-simplifier not installed (pip install onnxsim); skipping",
                file=sys.stderr,
            )

    metadata = {
        "application": "rvc-android",
        "version": "static-1.0",
        "modelType": "pyTorchRVCv2" if use_v2 else "pyTorchRVC",
        "samplingRate": sr,
        "f0": f0,
        "embChannels": emb_channels,
        "embedder": "hubert_base",
        "embOutputLayer": emb_layer,
        "useFinalProj": use_final_proj,
        "staticT": T,
    }
    meta = model.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps(metadata)
    onnx.save(model, args.output)

    print(f"wrote {args.output}")
    print(f"embedded metadata: {json.dumps(metadata)}")
    print(
        f"\nNext: copy this file to /sdcard/Download/RVC/ and pick it as the\n"
        f"Synthesizer in the app. The runtime will need matching chunking\n"
        f"(input audio is sliced so each synth call sees exactly T={T} frames)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
