#!/usr/bin/env python3
"""Render linear and dB range-time intensity plots from radar I/Q."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(image: np.ndarray, title: str, destination: Path, label: str) -> None:
    vmin = np.nanpercentile(image, 10)
    vmax = np.nanpercentile(image, 99)
    plt.figure(figsize=(12, 6))
    plt.imshow(image.T, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Time (frame index)")
    plt.ylabel("Range bin")
    plt.colorbar(label=label)
    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("radar_root", type=Path)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--drop-near", type=int, default=4)
    parser.add_argument("--drop-far", type=int, default=8)
    parser.add_argument("--no-clutter-removal", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    with (args.radar_root / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_chirps, num_tx, num_rx, num_samples = meta["iq"]["shape"]
    if not 0 <= args.tx < num_tx:
        raise IndexError(f"tx index {args.tx} outside [0, {num_tx})")

    elements_per_frame = num_chirps * num_tx * num_rx * num_samples * 2
    raw = np.memmap(args.radar_root / "iq", dtype=np.int16, mode="r")
    total_frames = len(raw) // elements_per_frame
    frame_count = min(total_frames, args.max_frames)
    range_bins = num_samples // 2
    profiles = np.zeros((frame_count, range_bins), dtype=np.float32)
    window = np.hamming(num_samples).astype(np.float32)

    for frame_index in range(frame_count):
        start = frame_index * elements_per_frame
        frame = np.asarray(raw[start : start + elements_per_frame], dtype=np.float32)
        frame = frame.reshape(num_chirps, num_tx, num_rx, num_samples, 2)
        signal = frame[:, args.tx, :, :, 0] + 1j * frame[:, args.tx, :, :, 1]
        if not args.no_clutter_removal:
            signal -= signal.mean(axis=0, keepdims=True)
        spectrum = np.fft.fft(signal * window, axis=-1)[..., :range_bins]
        profiles[frame_index] = np.abs(spectrum).mean(axis=(0, 1))

    cleaned = profiles.copy()
    cleaned[:, : args.drop_near] = 1e-6
    if args.drop_far:
        cleaned[:, -args.drop_far :] = 1e-6
    decibels = 20.0 * np.log10(cleaned + 1e-6)
    output_dir = args.output_dir or args.radar_root / "range_profile_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"T{frame_count}_tx{args.tx}"
    render(profiles, f"RTI (linear) — {suffix}", output_dir / f"rti_linear_{suffix}.jpg", "Magnitude")
    render(decibels, f"RTI (dB) — {suffix}", output_dir / f"rti_db_{suffix}.jpg", "dB")
    print(output_dir)


if __name__ == "__main__":
    main()

