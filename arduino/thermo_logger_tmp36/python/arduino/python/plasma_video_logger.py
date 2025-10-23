#!/usr/bin/env python3
"""
plasma_video_logger.py

Capture video brightness from a Region of Interest (ROI) AND (optionally) read
temperature from an Arduino over serial. Writes two time-stamped CSVs you can
sync later during analysis.

Usage examples:
  # Video file only (from iPhone after ffmpeg prep)
  python python/plasma_video_logger.py --video videos/run1_fixed.mp4 --out results/runs/run1

  # USB camera + Arduino TMP36 on COM3 (Windows) or /dev/ttyACM0 (Linux/Mac)
  python python/plasma_video_logger.py --camera 0 --serial COM3 --baud 115200 --out results/runs/liverun

Press 'q' in the preview window to stop.
"""

import argparse, csv, os, sys, time, threading, queue
from pathlib import Path

import cv2
import numpy as np

try:
    import serial  # pyserial
except Exception:
    serial = None


# ------------------------- Helpers -------------------------

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def frac_roi_to_px(h, w, roi_frac):
    """roi_frac = (xc, yc, ww, hh) in 0..1 -> (x0,y0,x1,y1) ints"""
    xc, yc, ww, hh = roi_frac
    W, H = int(ww * w), int(hh * h)
    x0 = max(0, int(xc * w - W // 2)); y0 = max(0, int(yc * h - H // 2))
    x1 = min(w, x0 + W); y1 = min(h, y0 + H)
    return x0, y0, x1, y1


# ------------------------- Serial reader -------------------------

def serial_worker(port: str, baud: int, out_csv_path: str, stop_event: threading.Event):
    """
    Reads lines like:  T,24.37
    Writes CSV columns: t_s, temp_C
    """
    if serial is None:
        print("[WARN] pyserial not installed. Skipping serial logging.")
        return
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=1)
    except Exception as e:
        print(f"[WARN] Could not open serial port {port}: {e}")
        return

    ensure_dir(out_csv_path)
    t0 = time.time()
    with open(out_csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "temp_C"])
        while not stop_event.is_set():
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                line = ""
            if not line:
                continue
            # Expected "T,24.12"
            if line.startswith("T,"):
                try:
                    val = float(line.split(",")[1])
                    wr.writerow([time.time() - t0, f"{val:.3f}"])
                except Exception:
                    pass
    try:
        ser.close()
    except Exception:
        pass


# ------------------------- Main video logger -------------------------

def run(args):
    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(int(args.camera))

    if not cap.isOpened():
        print("[ERR] Could not open video source.")
        sys.exit(1)

    # Read first frame to define ROI
    ok, frame = cap.read()
    if not ok:
        print("[ERR] No frames in source.")
        sys.exit(1)

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = frac_roi_to_px(h, w, (args.roi_xc, args.roi_yc, args.roi_w, args.roi_h))

    # Prepare output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_csv = out_dir / "video_intensity.csv"
    ser_csv = out_dir / "temp_serial.csv"

    # Start serial thread if requested
    stop_event = threading.Event()
    if args.serial:
        threading.Thread(
            target=serial_worker,
            args=(args.serial, args.baud, str(ser_csv), stop_event),
            daemon=True,
        ).start()
        print(f"[INFO] Serial logging -> {ser_csv} (port={args.serial}, baud={args.baud})")

    # Logging loop
    print(f"[INFO] Video ROI (px): x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(f"[INFO] Video intensity log -> {vid_csv}")
    t0 = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    write_header = True

    with open(vid_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "mean_intensity", "std_intensity", "roi_x0", "roi_y0", "roi_x1", "roi_y1"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y0:y1, x0:x1]
            meanI = float(roi.mean())
            stdI = float(roi.std())
            wr.writerow([time.time() - t0, f"{meanI:.3f}", f"{stdI:.3f}", x0, y0, x1, y1])

            if args.display:
                # Draw ROI and show mean
                disp = frame.copy()
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(disp, f"I={meanI:.1f}", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Plasma Video Logger (press q to quit)", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    print("[DONE] Logging finished.")


# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plasma video + temperature logger")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to MP4/MOV video (e.g., ffmpeg-converted iPhone clip)")
    src.add_argument("--camera", type=int, help="OpenCV camera index (e.g., 0)")

    p.add_argument("--serial", type=str, default=None, help="Serial port for Arduino (e.g., COM3 or /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default 115200)")
    p.add_argument("--out", type=str, required=True, help="Output folder for CSV logs")
    p.add_argument("--fps", type=float, default=60.0, help="Fallback FPS if video metadata missing")
    p.add_argument("--roi_xc", type=float, default=0.5, help="ROI center x (0..1)")
    p.add_argument("--roi_yc", type=float, default=0.5, help="ROI center y (0..1)")
    p.add_argument("--roi_w", type=float, default=0.4, help="ROI width fraction (0..1)")
    p.add_argument("--roi_h", type=float, default=0.4, help="ROI height fraction (0..1)")
    p.add_argument("--display", action="store_true", help="Show live preview with ROI box")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
