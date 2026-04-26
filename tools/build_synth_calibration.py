#!/usr/bin/env python3
"""Build a Synth W8A16 calibration corpus by running HuBERT and RMVPE
ORT sessions on the audio_30720 windows produced by
tools/build_calibration_corpus.py.

The synth ONNX takes 6 inputs:

  feats       FP16 [1, 192, 768]   ← HuBERT 6×5120 → concat → upsample 2× → pad
  p_len       INT64 [1]            ← constant 192
  pitch       INT64 [1, 192]       ← melQuantize(pitchf[:192])
  pitchf      FP32 [1, 192]        ← RMVPE @ 30720 → truncate 193→192
  sid         INT64 [1]            ← 0
  rand_noise  FP16 [1, 192, 192]   ← Gaussian (PRNG, seeded for reproducibility)

The feats path mirrors RvcPipeline.kt's perChunkHubert branch:
  6 × 5120 sub-audio → ContentVec → 6 × 15 frames → concat → 90 frames @ 50 Hz
  → upsample2xNearest → 180 frames @ 100 Hz → pad to 192 by replicating
  the last valid frame.

Output is a single .npz with one array per input name, packed as
(N_samples, *input_shape) so AI Hub's calibration_data dict can be
built by indexing along axis 0.

Usage
-----
    python tools/build_synth_calibration.py \\
        --audio-corpus qnn_build/calibration/audio_corpus.npz \\
        --hubert-onnx C:/Users/hurwy/Downloads/_/MMVCServerSIO/pretrain/content_vec_500.onnx \\
        --rmvpe-onnx  C:/Users/hurwy/Downloads/_/MMVCServerSIO/pretrain/rmvpe.onnx \\
        --out qnn_build/calibration/synth_corpus.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np


# Match RvcPipeline.kt
HUBERT_SUB_LEN = 5120
HUBERT_FRAMES_PER_SUB = HUBERT_SUB_LEN // 320 - 1  # ContentVec receptive field
CHUNK_SAMPLES = 30720
STATIC_T = 192
EMB_CHANNELS = 768
INTER_CHANNELS = 192  # rand_noise channel dim
RMVPE_THRESHOLD = 0.3

# melQuantize constants — mirror RmvpePitchExtractor.kt exactly so the
# pitch indices we feed to the synth match what the on-device pipeline
# would produce.
F0_MIN = 50.0
F0_MAX = 1100.0
F0_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
F0_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)


def mel_quantize(pitchf: np.ndarray) -> np.ndarray:
    """Vectorised port of RmvpePitchExtractor.kt's melQuantize."""
    span = F0_MEL_MAX - F0_MEL_MIN
    out = np.ones_like(pitchf, dtype=np.float64)
    voiced = pitchf > 0
    mel = 1127.0 * np.log(1.0 + pitchf[voiced] / 700.0)
    scaled = (mel - F0_MEL_MIN) * 254.0 / span + 1.0
    scaled = np.where(mel > 0, scaled, 1.0)
    out[voiced] = scaled
    out = np.clip(out, 1.0, 255.0)
    return np.round(out).astype(np.int64)


def upsample2x_nearest(feats: np.ndarray) -> np.ndarray:
    """50 Hz → 100 Hz by frame duplication. Shape [T, C] → [2T, C]."""
    return np.repeat(feats, 2, axis=0)


def pad_to_static_t(feats_2x: np.ndarray, static_t: int) -> np.ndarray:
    """Match sliceFeatsToStaticT's tail behaviour: replicate the last
    valid frame to fill up to static_t."""
    t, c = feats_2x.shape
    if t == static_t:
        return feats_2x
    if t > static_t:
        return feats_2x[:static_t]
    out = np.empty((static_t, c), dtype=feats_2x.dtype)
    out[:t] = feats_2x
    out[t:] = feats_2x[-1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio-corpus", required=True,
                        help="path to .npz from build_calibration_corpus.py "
                             "(must contain audio_30720)")
    parser.add_argument("--hubert-onnx", required=True,
                        help="path to ContentVec / HuBERT ONNX (FP32, dynamic T)")
    parser.add_argument("--rmvpe-onnx", required=True,
                        help="path to RMVPE ONNX (FP32, dynamic T)")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--hubert-output", default="unit12",
                        help="ContentVec output tensor name (default unit12 for v2)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for rand_noise PRNG")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("onnxruntime not installed — pip install onnxruntime")

    audio_npz = Path(args.audio_corpus)
    if not audio_npz.is_file():
        sys.exit(f"audio corpus not found: {audio_npz}")
    npz = np.load(str(audio_npz))
    if "audio_30720" not in npz.files:
        sys.exit(f"audio corpus is missing 'audio_30720' (has: {npz.files})")
    audio_30720 = npz["audio_30720"]
    print(f"audio corpus: {audio_30720.shape} from {audio_npz}")

    hubert_path = Path(args.hubert_onnx)
    rmvpe_path = Path(args.rmvpe_onnx)
    if not hubert_path.is_file():
        sys.exit(f"hubert onnx not found: {hubert_path}")
    if not rmvpe_path.is_file():
        sys.exit(f"rmvpe onnx not found: {rmvpe_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"loading {hubert_path.name}…")
    hubert = ort.InferenceSession(str(hubert_path), sess_opts,
                                  providers=["CPUExecutionProvider"])
    print(f"loading {rmvpe_path.name}…")
    rmvpe = ort.InferenceSession(str(rmvpe_path), sess_opts,
                                 providers=["CPUExecutionProvider"])

    hubert_input_name = hubert.get_inputs()[0].name  # 'audio'
    hubert_outputs = [o.name for o in hubert.get_outputs()]
    if args.hubert_output not in hubert_outputs:
        sys.exit(f"hubert output '{args.hubert_output}' not found "
                 f"(have: {hubert_outputs})")
    rmvpe_input_names = [i.name for i in rmvpe.get_inputs()]
    print(f"hubert: input={hubert_input_name}, picking output={args.hubert_output}")
    print(f"rmvpe: inputs={rmvpe_input_names}")

    rng = np.random.default_rng(args.seed)
    n = len(audio_30720)
    feats_arr = np.empty((n, 1, STATIC_T, EMB_CHANNELS), dtype=np.float16)
    p_len_arr = np.empty((n, 1), dtype=np.int64)
    pitch_arr = np.empty((n, 1, STATIC_T), dtype=np.int64)
    pitchf_arr = np.empty((n, 1, STATIC_T), dtype=np.float32)
    sid_arr = np.zeros((n, 1), dtype=np.int64)
    rand_noise_arr = np.empty((n, 1, INTER_CHANNELS, STATIC_T), dtype=np.float16)

    thr = np.array([RMVPE_THRESHOLD], dtype=np.float32)

    for i, audio in enumerate(audio_30720):
        # ---- HuBERT path: 6 × 5120 → 90 frames → upsample 2× → pad to 192
        sub_feats = []
        for s in range(CHUNK_SAMPLES // HUBERT_SUB_LEN):
            sub = audio[s * HUBERT_SUB_LEN:(s + 1) * HUBERT_SUB_LEN]
            sub = sub.reshape(1, HUBERT_SUB_LEN).astype(np.float32)
            out = hubert.run([args.hubert_output], {hubert_input_name: sub})[0]
            # out shape: [1, T_frames, 768] fp32 — squeeze batch
            sub_feats.append(out[0])
        feats_50hz = np.concatenate(sub_feats, axis=0)  # [90, 768]
        feats_100hz = upsample2x_nearest(feats_50hz)     # [180, 768]
        feats_192 = pad_to_static_t(feats_100hz, STATIC_T)
        feats_arr[i, 0] = feats_192.astype(np.float16)

        # ---- RMVPE path: 30720 → 193 frames → truncate to 192
        wave = audio.reshape(1, CHUNK_SAMPLES).astype(np.float32)
        pitchf_193 = rmvpe.run(None, {
            rmvpe_input_names[0]: wave,
            rmvpe_input_names[1]: thr,
        })[0]
        # rmvpe output may be [193] or [1, 193] depending on graph
        pf = pitchf_193.reshape(-1)
        if pf.shape[0] < STATIC_T:
            sys.exit(f"rmvpe returned {pf.shape[0]} frames, need ≥ {STATIC_T}")
        pf_192 = pf[:STATIC_T].astype(np.float32)
        pitchf_arr[i, 0] = pf_192
        pitch_arr[i, 0] = mel_quantize(pf_192)

        p_len_arr[i, 0] = STATIC_T
        rand_noise_arr[i, 0] = rng.standard_normal(
            (INTER_CHANNELS, STATIC_T), dtype=np.float32,
        ).astype(np.float16)

        if (i + 1) % 10 == 0 or i == n - 1:
            voiced = int((pitch_arr[i, 0] > 1).sum())
            print(f"  [{i+1}/{n}] voiced={voiced}/{STATIC_T} "
                  f"pitchf range=[{pf_192.min():.1f}, {pf_192.max():.1f}] Hz")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        feats=feats_arr,
        p_len=p_len_arr,
        pitch=pitch_arr,
        pitchf=pitchf_arr,
        sid=sid_arr,
        rand_noise=rand_noise_arr,
    )
    print()
    print(f"feats        {feats_arr.shape} {feats_arr.dtype}")
    print(f"p_len        {p_len_arr.shape} {p_len_arr.dtype}")
    print(f"pitch        {pitch_arr.shape} {pitch_arr.dtype}  "
          f"voiced/total = {int((pitch_arr > 1).sum())}/{pitch_arr.size}")
    print(f"pitchf       {pitchf_arr.shape} {pitchf_arr.dtype}")
    print(f"sid          {sid_arr.shape} {sid_arr.dtype}")
    print(f"rand_noise   {rand_noise_arr.shape} {rand_noise_arr.dtype}")
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
