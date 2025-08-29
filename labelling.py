#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import cv2

LABEL_MAP = {-1: "UNLABELED", 0: "NO_POINT", 1: "IN_POINT"}

HELP_TEXT = """\
Controls:
  p            set current label to IN_POINT
  n            set current label to NO_POINT
  u            set current label to UNLABELED (do not write labels)
  SPACE        pause / resume
  d or .       (while paused) step forward 1 frame
  a or ,       (while paused) step back 1 frame
  [ and ]      decrease / increase playback speed
  s            save labels to disk
  h            show/hide this help
  q or ESC     save & quit

Notes:
- The current label is applied automatically to each frame as the video plays.
- Labels are saved as a 1D NumPy array of shape [num_frames] with values in {-1,0,1}.
- Files are written to the output directory with the same basename as the video:
    <outdir>/<video_basename>.labels.npy  (labels array)
    <outdir>/<video_basename>.labels.json (metadata)
"""

def human_time(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return "?:??:??"
    secs = int(max(0, seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def save_labels(outdir, base, labels, meta):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"{base}.labels.npy"), labels.astype(np.int8))
    with open(os.path.join(outdir, f"{base}.labels.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {os.path.join(outdir, base+'.labels.npy')}")
    print(f"[saved] {os.path.join(outdir, base+'.labels.json')}")

def main():
    parser = argparse.ArgumentParser(
        description="Play a tennis match video (default 2x) and label frames as IN_POINT ('p') or NO_POINT ('n')."
    )
    parser.add_argument("video", help="Path to the input .mp4 (or any OpenCV-readable) video")
    parser.add_argument("--outdir", default="data", help="Output directory for labels (default: data)")
    parser.add_argument("--speed", type=float, default=2.0, help="Initial playback speed multiplier (default: 2.0x)")
    parser.add_argument("--window", default="Tennis Labeler", help="Window title (default: Tennis Labeler)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"File not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("[warn] FPS not reported by the container; assuming 30 FPS")
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(os.path.basename(args.video))[0]
    outdir = args.outdir

    # Load or initialize labels
    labels_path = os.path.join(outdir, f"{base}.labels.npy")
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        if labels.shape[0] != total_frames:
            print(f"[warn] Existing labels length ({labels.shape[0]}) != video frames ({total_frames}); reinitializing.")
            labels = np.full(total_frames, -1, dtype=np.int8)
        else:
            print(f"[info] Loaded existing labels from {labels_path}")
    else:
        labels = np.full(total_frames, -1, dtype=np.int8)

    # Metadata (updated on each save)
    meta = {
        "video": os.path.abspath(args.video),
        "frame_count": total_frames,
        "fps": fps,
        "size": {"width": width, "height": height},
        "labels": {"-1": "UNLABELED", "0": "NO_POINT", "1": "IN_POINT"},
        "created": datetime.utcnow().isoformat() + "Z",
        "last_saved": None,
        "tool": "tennis_point_labeler.py",
        "notes": "Labels are per-frame; apply current label as playback advances. Unlabeled frames are -1."
    }

    window = args.window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Playback & labeling state
    current_label = -1  # -1 unlabeled, 0 no point, 1 in point
    paused = False
    show_help = True
    speed = max(0.1, float(args.speed))
    delay_ms = max(1, int(1000.0 / (fps * speed)))

    # Position
    frame_idx = 0
    need_new_frame = True
    current_frame = None

    # Attempt to seek to first unlabeled frame if any
    if np.any(labels == -1):
        first_unlabeled = int(np.argmax(labels == -1))
        frame_idx = first_unlabeled
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        need_new_frame = True
        print(f"[info] Starting at first unlabeled frame: {frame_idx}/{total_frames}")
    else:
        print("[info] All frames already labeled. You can revise with pause/step-back.")

    last_save_time = time.time()

    def redraw_overlay(img, label_val, idx, fps_val, spd, paused_flag):
        overlay = img.copy()
        # HUD text
        label_name = LABEL_MAP.get(label_val, "UNLABELED")
        color = (0, 200, 0) if label_val == 1 else ((0, 0, 200) if label_val == 0 else (80, 80, 80))
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.putText(img, f"Label: {label_name}   (p: IN_POINT, n: NO_POINT, u: UNLABELED)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        t_cur = idx / fps_val if fps_val > 0 else 0.0
        t_total = total_frames / fps_val if fps_val > 0 else 0.0
        status = "PAUSED" if paused_flag else "PLAY"
        cv2.putText(img, f"{status}  |  Frame {idx+1}/{total_frames}  |  {human_time(t_cur)} / {human_time(t_total)}  |  {spd:.2f}x",
                    (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

        if show_help:
            y0 = 100
            for i, line in enumerate(HELP_TEXT.splitlines()[:12]):  # show first dozen lines
                cv2.putText(img, line, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        return img

    print(HELP_TEXT)
    print("[info] Focus the video window to use the hotkeys.")

    while True:
        if need_new_frame:
            # End if we've reached the last frame
            if frame_idx >= total_frames:
                print("[info] Reached end of video.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[warn] Failed to read frame; ending.")
                break

            current_frame = frame
            # Apply current label to this frame
            if current_label in (-1, 0, 1):
                labels[frame_idx] = current_label

            frame_idx += 1
            # If paused, don't advance automatically on next loop
            need_new_frame = not paused

        display = current_frame.copy() if current_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        display = redraw_overlay(display, current_label, frame_idx-1, fps, speed, paused)
        cv2.imshow(window, display)

        # Wait for key; use a small wait while paused for responsiveness
        wait = 50 if paused else delay_ms
        key = cv2.waitKey(wait) & 0xFF

        if key == 255:  # no key
            continue

        if key in (ord('q'), 27):  # q or ESC
            meta["last_saved"] = datetime.utcnow().isoformat() + "Z"
            save_labels(outdir, base, labels, meta)
            print("[info] Exiting.")
            break
        elif key == ord('p'):
            current_label = 1
        elif key == ord('n'):
            current_label = 0
        elif key == ord('u'):
            current_label = -1
        elif key == ord(' '):  # pause/resume
            paused = not paused
            need_new_frame = not paused
        elif key in (ord('d'), ord('.')):  # step forward 1 frame (while paused)
            if paused:
                need_new_frame = True
        elif key in (ord('a'), ord(',')):  # step back 1 frame (while paused)
            if paused:
                # Step back: go to previous frame (two back because we increment after writing)
                frame_idx = max(0, frame_idx - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                need_new_frame = True
        elif key == ord('s'):
            meta["last_saved"] = datetime.utcnow().isoformat() + "Z"
            save_labels(outdir, base, labels, meta)
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord(']'):
            speed = min(16.0, speed * 1.5)
            delay_ms = max(1, int(1000.0 / (fps * speed)))
            print(f"[speed] {speed:.2f}x")
        elif key == ord('['):
            speed = max(0.1, speed / 1.5)
            delay_ms = max(1, int(1000.0 / (fps * speed)))
            print(f"[speed] {speed:.2f}x")

        # Autosave every ~30 seconds
        if time.time() - last_save_time > 30:
            meta["last_saved"] = datetime.utcnow().isoformat() + "Z"
            save_labels(outdir, base, labels, meta)
            last_save_time = time.time()

    # Final save on exit (in case loop ended by EOF)
    meta["last_saved"] = datetime.utcnow().isoformat() + "Z"
    save_labels(outdir, base, labels, meta)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()