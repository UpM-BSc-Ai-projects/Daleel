import cv2
import pandas as pd
import time
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import RTDETR

# Import boxmot's create_tracker
try:
    from boxmot.trackers.tracker_zoo import create_tracker
    from boxmot.utils import TRACKER_CONFIGS
except ImportError:
    print("Error: 'boxmot' library imports failed.")
    print("Please ensure boxmot is installed: pip install boxmot")
    create_tracker = None

# Configuration
# VIDEO_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos" 
# Updating to user's likely path or keeping it configurable. The user previously had 'Result/Test' or similar. 
# Reverting to the one in previous user edit:
VIDEO_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos"
OUTPUT_DIR = "tracking_results"
MODEL_PATH = "rtdetr-l.pt" 
PROCESS_DURATION_SEC = 30

def get_boxmot_tracker(tracker_type, device):
    """
    Wrapper to create tracker using boxmot's factory.
    Loads default config and applies crowd-specific overrides.
    """
    if create_tracker is None:
        raise ImportError("BoxMOT/Tracking library not correctly installed.")

    # ReID weights path - BoxMOT often expects a path.
    # We'll use a standard lightweight one.
    reid_weights = Path("osnet_x0_25_msmt17.pt")
    
    # Mapping user request to boxmot keys
    # boxmot keys: ocsort, bytetrack, deepocsort
    # DeepSORT removed as requested.
    
    mapping = {
        'ByteTrack': 'bytetrack',
        'OC-SORT': 'ocsort',
        'DeepOCSORT': 'deepocsort'
    }
    
    boxmot_type = mapping.get(tracker_type, tracker_type.lower())
    
    if boxmot_type not in mapping.values():
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

    tracker_config_path = TRACKER_CONFIGS / f"{boxmot_type}.yaml"
    
    # Load default config to modifying it
    import yaml
    with open(tracker_config_path, "r") as f:
        y = yaml.safe_load(f)
    
    # Extract default values
    # The yaml usually has structure {param: {default: val, help: ..., ...}}
    params = {k: v['default'] for k, v in y.items()}
    
    # --- CROWD TUNING ---
    # Increase max_age / track_buffer to handle occlusions better
    # Standard is ~30 frames (1 sec). For crowds, people can be hidden for 2-3s.
    if 'max_age' in params: 
        params['max_age'] = 60 
    if 'track_buffer' in params: # ByteTrack equivalent
        params['track_buffer'] = 60
    
    # Adjust matching thresholds if needed
    # For crowds, strict matching helps avoid ID switches but might fragment tracks.
    # keeping defaults (usually 0.8 IoU) as they are robust.
    
    # Initialize tracker with modified params
    track = create_tracker(
        tracker_type=boxmot_type,
        tracker_config=tracker_config_path, 
        reid_weights=reid_weights,
        device=device,
        half=False,
        per_class=False,
        evolve_param_dict=params
    )
    return track

def process_video(video_path, tracker_name, model, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_to_process = int(fps * PROCESS_DURATION_SEC)
    
    # Setup Output Video
    video_name = Path(video_path).stem
    out_path = os.path.join(OUTPUT_DIR, f"output_{video_name}_{tracker_name}.mp4")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize Tracker
    try:
        tracker = get_boxmot_tracker(tracker_name, device)
    except Exception as e:
        print(f"Failed to init tracker {tracker_name}: {e}")
        return None
    
    # Validation Data
    track_history = {} 
    unique_ids = set()
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing {video_name} with {tracker_name}...")

    while cap.isOpened() and frame_count < total_frames_to_process:
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8 Inference
        results = model.predict(frame, conf=0.45, verbose=False, device=device, classes=[0])
        det_results = results[0].boxes.data.cpu().numpy() # x1, y1, x2, y2, conf, cls
        
        # Tracker Update
        # BoxMOT update expects: detections (x1, y1, x2, y2, conf, cls), image
        if len(det_results) > 0:
            tracks = tracker.update(det_results, frame)
        else:
            # Handle empty detections
            # Some trackers might require a different call or just passing empty array
            # BoxMOT's update typically handles empty, but let's check.
            # Usually passing empty array `np.empty((0, 6))` works.
            tracks = tracker.update(np.empty((0, 6)), frame)
            
        # tracks output: [x1, y1, x2, y2, id, conf, cls, ind] (usually)
        
        if len(tracks) > 0:
             for track in tracks:
                # Ensure we have enough values
                if len(track) < 5: continue
                
                try:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                except Exception:
                    continue

                unique_ids.add(track_id)
                
                # Update stats
                if track_id not in track_history:
                    track_history[track_id] = {'count': 1}
                else:
                    track_history[track_id]['count'] += 1

                # Draw
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames_to_process} frames")
    
    end_time = time.time()
    cap.release()
    out.release()
    
    # Calculate Metrics
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0
    total_unique_ids = len(unique_ids)
    
    if total_unique_ids > 0:
        total_len = sum(d['count'] for d in track_history.values())
        avg_track_len = total_len / total_unique_ids
    else:
        avg_track_len = 0
        
    frag_index = (total_unique_ids / frame_count) if frame_count > 0 else 0

    metrics = {
        "Video": video_name,
        "Tracker": tracker_name,
        "Inference Speed (FPS)": round(avg_fps, 2),
        "Total Unique IDs": total_unique_ids,
        "Avg Track Length": round(avg_track_len, 2),
        "Fragmentation Index": round(frag_index, 4)
    }
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        # Allow ultralytics to auto-download if missing logic handled by YOLO constructor usually
        pass
     # Load RT-DETR model
    model = RTDETR(MODEL_PATH)
    
    # Trackers
    # DeepSORT removed for crowd optimization focus
    tracker_types = ["ByteTrack", "OC-SORT", "DeepOCSORT"]
    
    # Videos
    video_files = [f"vid_{i}.mp4" for i in range(9, 14)]
    
    results = []

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"Warning: Video not found at {video_path}")
            continue
            
        for tracker_type in tracker_types:
            try:
                metrics = process_video(video_path, tracker_type, model, device)
                if metrics:
                    results.append(metrics)
            except Exception as e:
                print(f"Error processing {video_file} with {tracker_type}: {e}")
                # Print stack trace if helpful
                import traceback
                traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        print("\n=== Tracker Evaluation Results ===")
        print(df.to_string())
        
        csv_path = os.path.join(OUTPUT_DIR, "tracker_evaluation_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
