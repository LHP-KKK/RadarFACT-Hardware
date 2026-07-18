#!/usr/bin/env python3
"""Render a range profile from one frame of an I/Q-1M-like radar scene."""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("radar_root", type=Path, help="Directory containing iq and meta.json")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--chirp", type=int, default=0)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--rx", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    iq_path = args.radar_root / "iq"
    meta_path = args.radar_root / "meta.json"
    output_dir = args.output_dir or args.radar_root / "range_profile_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    iq_shape = tuple(meta["iq"]["shape"])
    num_chirps, num_tx, num_rx, num_samples = iq_shape
    complex_elements = int(np.prod(iq_shape))
    elements_per_frame = complex_elements * 2

    if iq_path.is_dir():
        candidates = sorted(glob.glob(str(iq_path / "*.bin")))
        if not candidates:
            raise FileNotFoundError(f"No .bin file found in {iq_path}")
        target = Path(candidates[0])
    else:
        target = iq_path

    raw = np.memmap(target, dtype=np.int16, mode="r")
    total_frames = len(raw) // elements_per_frame
    if not 0 <= args.frame < total_frames:
        raise IndexError(f"frame {args.frame} outside [0, {total_frames})")
    if not 0 <= args.chirp < num_chirps or not 0 <= args.tx < num_tx or not 0 <= args.rx < num_rx:
        raise IndexError("chirp/tx/rx index outside configured shape")

    start = args.frame * elements_per_frame
    frame = np.asarray(raw[start : start + elements_per_frame]).reshape(*iq_shape, 2)
    signal = frame[..., 0].astype(np.float32) + 1j * frame[..., 1].astype(np.float32)
    channel = signal[args.chirp, args.tx, args.rx].copy()
    channel -= np.mean(channel)
    magnitude = np.abs(np.fft.fft(channel * np.hamming(num_samples)))[: num_samples // 2]

    plt.figure(figsize=(10, 5))
    plt.plot(magnitude, linewidth=1)
    plt.fill_between(range(len(magnitude)), magnitude, alpha=0.3)
    plt.title(f"Range profile — frame {args.frame}")
    plt.xlabel("Range bin")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    destination = output_dir / f"frame_{args.frame:06d}_range_profile.jpg"
    plt.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close()
    print(destination)


if __name__ == "__main__":
    main()

