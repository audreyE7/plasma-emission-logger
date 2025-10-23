#!/usr/bin/env python3
"""
analyze_run.py

Load CSVs produced by plasma_video_logger.py, align timelines, and generate:
- intensity vs time
- temperature vs time (if available)
- scatter (intensity vs temperature) + Pearson r
- optional FFT of intensity (to see 60 Hz flicker)
Saves figures into results/figs and a summary JSON.

Usage:
  python python/analyze_run.py --run results/runs/run1 --fps 60
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv_maybe(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def align_and_interpolate(video_df, temp_df, fps):
    """
    Returns a single DataFrame with:
      t_s, I_mean, I_std, temp_C (if available)
    Temperature is interpolated onto the video timeline for correlation.
    """
    out = pd.DataFrame()
    out["t_s"] = video_df["t_s"].astype(float)
    out["I_mean"] = video_df["mean_intensity"].astype(float)
    out["I_std"]  = video_df["std_intensity"].astype(float)

    if temp_df is not None and "t_s" in temp_df.columns and "temp_C" in temp_df.columns:
        # Interpolate temperature onto video time base
        tT   = temp_df["t_s"].astype(float).values
        TT   = temp_df["temp_C"].astype(float).values
        tVid = out["t_s"].values
        T_interp = np.interp(tVid, tT, TT)
        out["temp_C"] = T_interp
    else:
        out["temp_C"] = np.nan
    return out

def compute_fft(time_s, x):
    # Uniform sampling assumed. If not uniform, this is still a good approx for visualization.
    dt = np.median(np.diff(time_s))
    x = x - np.mean(x)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(np.fft.rfft(x)) * 2.0 / n
    return freqs, mag

def main(args):
    run_dir = Path(args.run)
    figs_dir = Path("results/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    vid_csv  = run_dir / "video_intensity.csv"
    tmp_csv  = run_dir / "temp_serial.csv"
    video_df = load_csv_maybe(vid_csv)
    temp_df  = load_csv_maybe(tmp_csv)

    if video_df is None:
        raise SystemExit(f"Missing {vid_csv}")

    df = align_and_interpolate(video_df, temp_df, args.fps)

    # ---- Metrics
    metrics = {}
    metrics["duration_s"] = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0])
    if df["temp_C"].notna().sum() > 3:
        r = np.corrcoef(df["I_mean"], df["temp_C"])[0,1]
        metrics["pearson_I_vs_T"] = float(r)
    else:
        metrics["pearson_I_vs_T"] = None

    # ---- Plots
    # 1) Intensity vs time + (optional) temp
    fig1, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(df["t_s"], df["I_mean"], lw=1.5, label="Intensity (ROI mean)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Intensity (a.u.)")
    if df["temp_C"].notna().sum() > 3:
        ax2 = ax1.twinx()
        ax2.plot(df["t_s"], df["temp_C"], lw=1.0, alpha=0.7, label="Temp (°C)", color="tab:red")
        ax2.set_ylabel("Temp (°C)")
        ax1.legend(loc="upper left")
    fig1.tight_layout()
    p1 = figs_dir / f"{run_dir.name}_intensity_temp.png"
    fig1.savefig(p1, dpi=180)
    plt.close(fig1)

    # 2) Scatter: intensity vs temperature (if available)
    if df["temp_C"].notna().sum() > 3:
        fig2, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(df["temp_C"], df["I_mean"], s=6, alpha=0.6)
        rtxt = f"r = {metrics['pearson_I_vs_T']:.3f}" if metrics["pearson_I_vs_T"] is not None else "r = N/A"
        ax.set_title(f"Intensity vs Temp\n{rtxt}")
        ax.set_xlabel("Temp (°C)")
        ax.set_ylabel("Intensity (a.u.)")
        fig2.tight_layout()
        p2 = figs_dir / f"{run_dir.name}_scatter_I_vs_T.png"
        fig2.savefig(p2, dpi=180)
        plt.close(fig2)

    # 3) FFT of intensity (flicker)
    freqs, mag = compute_fft(df["t_s"].values, df["I_mean"].values)
    fig3, ax = plt.subplots(figsize=(9, 3))
    ax.plot(freqs, mag, lw=1.2)
    ax.set_xlim(0, min(200, freqs.max()))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Intensity spectrum (look for 60 Hz & harmonics)")
    fig3.tight_layout()
    p3 = figs_dir / f"{run_dir.name}_fft.png"
    fig3.savefig(p3, dpi=180)
    plt.close(fig3)

    # ---- Save summary
    summary = {
        "run": str(run_dir),
        "figures": [str(p1), str(p3)] + ([str(p2)] if (df["temp_C"].notna().sum() > 3) else []),
        "metrics": metrics,
        "samples": {
            "n_frames": int(len(df)),
            "has_temp": bool(df["temp_C"].notna().sum() > 3)
        }
    }
    with open(figs_dir / f"{run_dir.name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:")
    for p in summary["figures"]:
        print("  -", p)
    print("Summary:", figs_dir / f"{run_dir.name}_summary.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to results/runs/<runname>")
    ap.add_argument("--fps", type=float, default=60.0, help="Video FPS (used for FFT axis label)")
    args = ap.parse_args()
    main(args)
