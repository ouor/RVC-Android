#!/usr/bin/env python3
"""Build a calibration corpus for AI Hub W8A16 compile jobs.

Loads a directory of WAV files (any sample rate, mono or stereo), uses
ffmpeg to standardize to 16 kHz mono float32 PCM, then stride-extracts
1.92 s windows of two lengths:

  audio_30720  → for RMVPE (full 1.92 s @ 16 kHz) and as the audio source
                 used downstream by tools/build_synth_calibration.py
  audio_5120   → for the current HuBERT compile config (audio_samples=5120
                 due to V79 VTCM constraint at 30720)

Note on resampling: ffmpeg's default resampler (swresample, soxr-like) is
higher quality than the linear interp the on-device pipeline uses. For
calibration this is fine — the quantizer fits an activation distribution,
and a slightly cleaner input distribution is not harmful. We just don't
want to pre-distort the calibration audio in ways production won't see.

Windows that are mostly silent (RMS < threshold) are skipped, since voiced
content drives the activation distribution that quantization actually has
to fit. Speakers are round-robined so no single clip dominates.

Output is a single .npz that all three compile_*_aihub.py scripts can
load. Synth calibration also needs feats/pitch/pitchf — those are produced
later by tools/build_synth_calibration.py from the same audio_30720 set.

Usage
-----
    python tools/build_calibration_corpus.py \\
        --src-dir C:/Users/hurwy/Downloads/_/samples \\
        --out qnn_build/calibration/audio_corpus.npz \\
        --n-samples 50
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


HUBERT_SR = 16000
WIN_30720 = 30720   # 1.92 s @ 16 kHz, matches synth chunk
WIN_5120 = 5120     # current HuBERT compile config


def load_via_ffmpeg(path: Path, ffmpeg_bin: str) -> np.ndarray:
    """Decode any audio file → 16 kHz mono float32 numpy array."""
    cmd = [
        ffmpeg_bin, "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(HUBERT_SR),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed on {path.name}: {e.stderr.decode('utf-8', 'replace')}"
        ) from e
    return np.frombuffer(proc.stdout, dtype=np.float32)


def extract_windows(audio16k: np.ndarray, win_size: int, hop: int,
                    rms_floor: float) -> list[np.ndarray]:
    windows = []
    for start in range(0, audio16k.size - win_size + 1, hop):
        w = audio16k[start:start + win_size]
        rms = float(np.sqrt(np.mean(w * w) + 1e-12))
        if rms < rms_floor:
            continue
        windows.append(w.copy())
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src-dir", required=True,
                        help="directory of source audio files (any SR/format)")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="number of calibration windows to keep (default 50)")
    parser.add_argument("--hop-30720", type=int, default=15360,
                        help="stride for 30720 windows (default 15360 = 50%% overlap)")
    parser.add_argument("--hop-5120", type=int, default=2560,
                        help="stride for 5120 windows (default 2560 = 50%% overlap)")
    parser.add_argument("--rms-floor", type=float, default=0.005,
                        help="skip windows with RMS below this (default 0.005)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for round-robin shuffle (default 0)")
    parser.add_argument("--ext", default="wav,flac,mp3,ogg,m4a,webm,mkv",
                        help="comma-separated extensions to scan (default wav,flac,...)")
    parser.add_argument("--ffmpeg", default=None,
                        help="path to ffmpeg binary (default: from PATH)")
    args = parser.parse_args()

    ffmpeg_bin = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        sys.exit("ffmpeg not found on PATH — install ffmpeg or pass --ffmpeg")

    src = Path(args.src_dir)
    if not src.is_dir():
        sys.exit(f"src dir not found: {src}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    exts = {f".{e.strip().lower()}" for e in args.ext.split(",") if e.strip()}
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
    if not files:
        sys.exit(f"no audio files matching {sorted(exts)} in {src}")
    print(f"found {len(files)} audio files in {src}")

    per_clip_30720: list[list[np.ndarray]] = []
    per_clip_5120: list[list[np.ndarray]] = []
    total_dur = 0.0
    for f in files:
        a = load_via_ffmpeg(f, ffmpeg_bin)
        if a.size == 0:
            print(f"  {f.name}: empty after decode, skipping")
            continue
        # ffmpeg pipe output is a read-only buffer; copy before mutating.
        a = np.nan_to_num(a, copy=True)
        total_dur += a.size / HUBERT_SR
        w30 = extract_windows(a, WIN_30720, args.hop_30720, args.rms_floor)
        w5 = extract_windows(a, WIN_5120, args.hop_5120, args.rms_floor)
        per_clip_30720.append(w30)
        per_clip_5120.append(w5)
        print(f"  {f.name:50s} 16k_dur={a.size / HUBERT_SR:6.2f}s  "
              f"win30720={len(w30):3d}  win5120={len(w5):3d}")

    print(f"total resampled duration: {total_dur:.1f} s")

    rng = np.random.default_rng(args.seed)

    def round_robin(per_clip: list[list[np.ndarray]], n: int) -> np.ndarray:
        shuffled = []
        for clip_wins in per_clip:
            order = list(range(len(clip_wins)))
            rng.shuffle(order)
            shuffled.append([clip_wins[i] for i in order])
        merged: list[np.ndarray] = []
        i = 0
        while len(merged) < n:
            advanced = False
            for clip_wins in shuffled:
                if i < len(clip_wins):
                    merged.append(clip_wins[i])
                    advanced = True
                    if len(merged) >= n:
                        break
            if not advanced:
                break
            i += 1
        if len(merged) < n:
            print(f"warning: only {len(merged)} non-silent windows available "
                  f"(asked for {n})")
        return np.stack(merged) if merged else np.empty((0,))

    audio_30720 = round_robin(per_clip_30720, args.n_samples)
    audio_5120 = round_robin(per_clip_5120, args.n_samples)

    if audio_30720.size == 0 or audio_5120.size == 0:
        sys.exit("calibration extraction produced 0 windows — lower --rms-floor or "
                 "supply more audio")

    print()
    rms30 = np.sqrt((audio_30720 ** 2).mean(axis=1))
    rms5 = np.sqrt((audio_5120 ** 2).mean(axis=1))
    print(f"audio_30720 shape: {audio_30720.shape} dtype={audio_30720.dtype}")
    print(f"  per-window RMS: min={rms30.min():.4f} max={rms30.max():.4f} "
          f"mean={rms30.mean():.4f}")
    print(f"audio_5120 shape: {audio_5120.shape} dtype={audio_5120.dtype}")
    print(f"  per-window RMS: min={rms5.min():.4f} max={rms5.max():.4f} "
          f"mean={rms5.mean():.4f}")

    np.savez_compressed(str(out),
                        audio_30720=audio_30720,
                        audio_5120=audio_5120)
    print(f"\nwrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
