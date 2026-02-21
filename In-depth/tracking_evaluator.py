"""
Unified Tracking Evaluator
==========================
Runs ByteTrack (boxmot) and OC-SORT (boxmot) trackers on all
videos in a given directory using a YOLO person-detection model, then evaluates
each tracker on:
  1. Motion Consistency
  2. Appearance-Based Consistency
  3. Track Fragmentation & Quality

Usage:
    python tracking_evaluator.py
    python tracking_evaluator.py --video_dir /path/to/videos --duration 30
"""

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import time
import torch
import yaml
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# BoxMOT imports (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from boxmot.trackers.tracker_zoo import create_tracker
    from boxmot.utils import TRACKER_CONFIGS
    BOXMOT_AVAILABLE = True
except ImportError:
    print("[WARN] boxmot library not found. ByteTrack & OC-SORT will be skipped.")
    print("       Install with:  pip install boxmot")
    BOXMOT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO_DIR   = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos\test"
DEFAULT_MODEL_PATH  = "yolo_person_c0m_yv11.pt"
DEFAULT_OUTPUT_DIR  = "eval_results"
DEFAULT_DURATION    = 60         # seconds of each video to process
DETECTION_CONF      = 0.45        # YOLO confidence threshold for person class
HIST_BINS           = 64          # histogram bins per channel for appearance metric

# =============================================================================
#  TRACKER PARAMETER DEFINITIONS (every param commented)
# =============================================================================

def get_bytetrack_params() -> dict:
    """
    Returns a dict of **every** ByteTrack parameter with comments.
    These are the defaults from the boxmot / Ultralytics implementation.
    """
    return {
        # ── Detection thresholds ──────────────────────────────────────────
        "track_high_thresh": 0.15,
        # First-stage detection confidence threshold.  Detections above this
        # value are matched to existing tracks first.  Higher → cleaner
        # tracks but may miss low-confidence persons.

        "track_low_thresh": 0.03,
        # Second-stage threshold for low-confidence detections.  Detections
        # between track_low_thresh and track_high_thresh are used to recover
        # lost tracks.  Lower → more aggressive recovery.

        "new_track_thresh": 0.25,
        # Minimum confidence required to initialise a brand-new track from
        # an unmatched detection.  Higher → fewer false-positive tracks.

        # ── Track lifecycle ───────────────────────────────────────────────
        "track_buffer": 100,
        # Maximum number of frames a lost track is kept alive before
        # deletion.  Higher → better occlusion handling but risk of
        # ghost tracks and ID switches.

        # ── Association ───────────────────────────────────────────────────
        "match_thresh": 0.9,
        # IoU (or cost) threshold used to associate detections to tracks.
        # Pairs with IoU below this value are not matched.

        "fuse_score": True,
        # Whether to fuse the detection confidence score with the IoU cost
        # during matching.  True → stabilises matching for weak detections.
    }

# =============================================================================
#  TRACKER FACTORY FUNCTIONS
# =============================================================================

def create_bytetrack(device):
    """Create a BoxMOT ByteTrack tracker with explicit parameters."""
    if not BOXMOT_AVAILABLE:
        raise ImportError("boxmot is required for ByteTrack.")

    params = get_bytetrack_params()
    config_path = TRACKER_CONFIGS / "bytetrack.yaml"
    reid_weights = Path("osnet_x0_25_msmt17.pt")

    tracker = create_tracker(
        tracker_type="bytetrack",
        tracker_config=config_path,
        reid_weights=reid_weights,
        device=device,
        half=False,               # FP16 inference for ReID (False = FP32)
        per_class=False,           # Track all classes together
        evolve_param_dict=params,  # Override defaults with our explicit params
    )
    return tracker


# =============================================================================
#  METRIC HELPERS
# =============================================================================

def compute_iou(boxA, boxB):
    """IoU between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def compute_hsv_histogram(frame, box):
    """
    Compute a normalised HSV histogram for the region inside `box`.
    box = (x1, y1, x2, y2).  Returns a flat numpy array.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def evaluate_tracks(track_data: dict, total_frames: int) -> dict:
    """
    Given collected per-frame track data, compute all evaluation metrics.

    track_data: dict of track_id -> list of dicts, each dict:
        {'frame': int, 'box': (x1,y1,x2,y2), 'center': (cx,cy), 'hist': np.array|None}
    """
    metrics = {}

    num_tracks = len(track_data)
    if num_tracks == 0:
        return {
            "Total Unique IDs": 0,
            "Avg Track Length": 0,
            "Fragmentation Index": 0,
            "Short Track Ratio (%)": 0,
            "Longest Track (frames)": 0,
            "Avg Velocity Smoothness": 0,
            "Avg Direction Change (deg)": 0,
            "Avg Acceleration Smoothness": 0,
            "Avg Histogram Consistency": 0,
            "Avg IoU Consistency": 0,
        }

    track_lengths = []
    velocity_smoothness_list = []
    direction_change_list = []
    accel_smoothness_list = []
    hist_consistency_list = []
    iou_consistency_list = []

    for tid, entries in track_data.items():
        entries = sorted(entries, key=lambda e: e['frame'])
        track_len = len(entries)
        track_lengths.append(track_len)

        if track_len < 2:
            continue

        # ── Motion Consistency ────────────────────────────────────────
        centers = np.array([e['center'] for e in entries], dtype=np.float64)
        velocities = np.diff(centers, axis=0)  # (N-1, 2)
        speeds = np.linalg.norm(velocities, axis=1)

        # Velocity smoothness = std of speed
        velocity_smoothness_list.append(np.std(speeds))

        # Direction changes
        if len(velocities) >= 2:
            angles = []
            for i in range(len(velocities) - 1):
                v1 = velocities[i]
                v2 = velocities[i + 1]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                    angles.append(np.degrees(np.arccos(cos_a)))
            if angles:
                direction_change_list.append(np.mean(angles))

        # Acceleration smoothness
        if len(velocities) >= 2:
            accels = np.diff(velocities, axis=0)
            accel_mags = np.linalg.norm(accels, axis=1)
            accel_smoothness_list.append(np.std(accel_mags))

        # ── Appearance-Based Consistency ──────────────────────────────
        # Histogram consistency
        hists = [e['hist'] for e in entries if e['hist'] is not None]
        if len(hists) >= 2:
            correlations = []
            for i in range(len(hists) - 1):
                corr = cv2.compareHist(
                    hists[i].reshape(-1, 1).astype(np.float32),
                    hists[i + 1].reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
                correlations.append(corr)
            hist_consistency_list.append(np.mean(correlations))

        # IoU consistency (consecutive-frame IoU for same track)
        boxes = [e['box'] for e in entries]
        ious = []
        for i in range(len(boxes) - 1):
            ious.append(compute_iou(boxes[i], boxes[i + 1]))
        iou_consistency_list.append(np.mean(ious))

    # ── Track Fragmentation & Quality ─────────────────────────────────
    avg_track_len = np.mean(track_lengths) if track_lengths else 0
    short_tracks = sum(1 for l in track_lengths if l < 10)
    short_track_ratio = (short_tracks / num_tracks * 100) if num_tracks else 0
    longest = max(track_lengths) if track_lengths else 0
    frag_index = num_tracks / total_frames if total_frames > 0 else 0

    metrics["Total Unique IDs"]              = num_tracks
    metrics["Avg Track Length"]              = round(avg_track_len, 2)
    metrics["Fragmentation Index"]           = round(frag_index, 4)
    metrics["Short Track Ratio (%)"]         = round(short_track_ratio, 2)
    metrics["Longest Track (frames)"]        = longest

    # ── Aggregated motion metrics ─────────────────────────────────────
    metrics["Avg Velocity Smoothness"]       = round(np.mean(velocity_smoothness_list), 4) if velocity_smoothness_list else 0
    metrics["Avg Direction Change (deg)"]    = round(np.mean(direction_change_list), 2) if direction_change_list else 0
    metrics["Avg Acceleration Smoothness"]   = round(np.mean(accel_smoothness_list), 4) if accel_smoothness_list else 0

    # ── Aggregated appearance metrics ─────────────────────────────────
    metrics["Avg Histogram Consistency"]     = round(np.mean(hist_consistency_list), 4) if hist_consistency_list else 0
    metrics["Avg IoU Consistency"]           = round(np.mean(iou_consistency_list), 4) if iou_consistency_list else 0

    return metrics


# =============================================================================
#  VIDEO PROCESSING — BoxMOT trackers (ByteTrack / OC-SORT)
# =============================================================================

def process_video_boxmot(video_path, tracker_name, model, device, duration_sec, output_dir):
    """Run a BoxMOT tracker (ByteTrack or OC-SORT) on a single video."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * duration_sec)

    # Output video
    video_name = Path(video_path).stem
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_name}_{tracker_name}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Create tracker
    if tracker_name == "ByteTrack":
        tracker = create_bytetrack(device)
    else:
        raise ValueError(f"Unknown BoxMOT tracker: {tracker_name}")

    # Collection structures
    track_data = defaultdict(list)   # track_id -> [{frame, box, center, hist}, ...]
    frame_idx  = 0
    t_start    = time.time()

    print(f"  → {tracker_name} on {video_name} ({max_frames} frames)...")

    while cap.isOpened() and frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # Detection
        results = model.predict(frame, conf=DETECTION_CONF, verbose=False, device=device, classes=[0])
        dets = results[0].boxes.data.cpu().numpy()          # (N, 6): x1 y1 x2 y2 conf cls

        # Tracker update
        if len(dets) > 0:
            tracks = tracker.update(dets, frame)
        else:
            tracks = tracker.update(np.empty((0, 6)), frame)

        # Collect & draw
        if len(tracks) > 0:
            for t in tracks:
                if len(t) < 5:
                    continue
                x1, y1, x2, y2 = map(int, t[:4])
                tid = int(t[4])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                hist = compute_hsv_histogram(frame, (x1, y1, x2, y2))

                track_data[tid].append({
                    'frame': frame_idx,
                    'box': (x1, y1, x2, y2),
                    'center': (cx, cy),
                    'hist': hist,
                })

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    {frame_idx}/{max_frames} frames")

    elapsed = time.time() - t_start
    cap.release()
    writer.release()

    # Evaluate
    metrics = evaluate_tracks(track_data, frame_idx)
    metrics["Video"]   = video_name
    metrics["Tracker"] = tracker_name
    metrics["Inference Speed (FPS)"] = round(frame_idx / elapsed, 2) if elapsed > 0 else 0
    metrics["Frames Processed"] = frame_idx

    print(f"    Done. {frame_idx} frames in {elapsed:.1f}s ({metrics['Inference Speed (FPS)']} FPS)")
    return metrics

# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Tracking Evaluator")
    parser.add_argument("--video_dir", type=str, default=DEFAULT_VIDEO_DIR,
                        help="Directory containing .mp4 video files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to YOLO person-detection weights (.pt).")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for output videos and CSV results.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Seconds of each video to process.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    print(f"Videos : {args.video_dir}")
    print(f"Model  : {args.model}")
    print(f"Output : {args.output_dir}")
    print(f"Duration: {args.duration}s per video\n")

    # Load model
    model = YOLO(args.model)

    # Discover videos
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = sorted([
        f for f in Path(args.video_dir).iterdir()
        if f.suffix.lower() in video_extensions
    ])
    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} video(s):\n  " + "\n  ".join(v.name for v in video_files))

    # ── Print tracker parameters ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BYTETRACK PARAMETERS")
    print("=" * 70)
    for k, v in get_bytetrack_params().items():
        print(f"  {k:25s} = {v}")

    # ── Run trackers ──────────────────────────────────────────────────
    all_results = []

    for vpath in video_files:
        print(f"\n{'─' * 60}")
        print(f"Video: {vpath.name}")
        print(f"{'─' * 60}")

        # 1. ByteTrack
        if BOXMOT_AVAILABLE:
            try:
                m = process_video_boxmot(vpath, "ByteTrack", model, device,
                                         args.duration, args.output_dir)
                if m:
                    all_results.append(m)
            except Exception as e:
                print(f"  [ERROR] ByteTrack failed: {e}")
                import traceback; traceback.print_exc()

    # ── Results ───────────────────────────────────────────────────────
    if all_results:
        # Reorder columns for readability
        col_order = [
            "Video", "Tracker", "Frames Processed", "Inference Speed (FPS)",
            # Motion
            "Avg Velocity Smoothness", "Avg Direction Change (deg)", "Avg Acceleration Smoothness",
            # Appearance
            "Avg Histogram Consistency", "Avg IoU Consistency",
            # Fragmentation & Quality
            "Total Unique IDs", "Avg Track Length", "Fragmentation Index",
            "Short Track Ratio (%)", "Longest Track (frames)",
        ]
        df = pd.DataFrame(all_results)
        # Ensure all columns present (some might be missing on error)
        for c in col_order:
            if c not in df.columns:
                df[c] = None
        df = df[col_order]

        print("\n" + "=" * 100)
        print("  EVALUATION RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))

        csv_path = os.path.join(args.output_dir, "tracking_evaluation_results_ByteTrack3.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results were generated.")


if __name__ == "__main__":
    main()
