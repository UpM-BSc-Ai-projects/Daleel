import cv2
import pandas as pd
import time
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import RTDETR

# Configuration
VIDEO_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos"
OUTPUT_DIR = "opencv_results"
MODEL_PATH = "rtdetr-l.pt"
PROCESS_DURATION_SEC = 30
DETECTION_INTERVAL = 1 # Run detection every frame to keep trackers strictly up to date

def create_opencv_tracker(tracker_type):
    """
    Creates an OpenCV tracker object based on the type.
    """
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

def compute_iou(boxA, boxB):
    # box: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def process_video_opencv_with_detection(video_path, tracker_type, model, device):
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
    
    # Setup Output
    video_name = Path(video_path).stem
    video_out_dir = os.path.join(OUTPUT_DIR, "videos")
    txt_out_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(video_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)
    
    out_path = os.path.join(video_out_dir, f"{video_name}_{tracker_type}_det.mp4")
    txt_path = os.path.join(txt_out_dir, f"{video_name}_{tracker_type}_det.txt")
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing {video_name} with {tracker_type} (Detection-Corrected)...")
    
    # State
    # List of dicts: {'tracker': obj, 'id': int, 'bbox': (x,y,w,h), 'lost_frames': int, 'color': tuple}
    active_trackers = [] 
    next_id = 1
    
    frame_count = 0
    start_time = time.time()
    tracking_data = [] # For saving to txt
    
    while cap.isOpened() and frame_count < total_frames_to_process:
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Update existing trackers
        to_remove = []
        for i, trk in enumerate(active_trackers):
            success, bbox = trk['tracker'].update(frame)
            if success:
                trk['bbox'] = bbox
                trk['lost_frames'] = 0
            else:
                trk['lost_frames'] += 1
                # If lost for too long, mark for removal (soft delete)
                # But we might recover via detection, so keep for a bit?
                # For now, if tracker fails internal update, we mark it lost.
                pass 
        
        # 2. Run Detection (Periodic)
        if frame_count % DETECTION_INTERVAL == 0:
            results = model.predict(frame, conf=0.45, verbose=False, device=device, classes=[0])
            det_results = results[0].boxes.data.cpu().numpy() # x1, y1, x2, y2, conf, cls
            
            # Convert detections to (x, y, w, h)
            detections = []
            for det in det_results:
                x1, y1, x2, y2 = map(int, det[:4])
                conf = det[4]
                detections.append({'bbox': (x1, y1, x2-x1, y2-y1), 'conf': conf, 'matched': False})
            
            # 3. Associate Detections to Trackers (Greedy IoU)
            # Match highest IoU first
            for trk in active_trackers:
                best_iou = 0
                best_det_idx = -1
                
                for d_idx, det in enumerate(detections):
                    if det['matched']: continue
                    
                    iou = compute_iou(trk['bbox'], det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = d_idx
                
                # Threshold for association
                if best_iou > 0.3:
                    # Match found! 
                    # Correct tracker with detection box
                    det = detections[best_det_idx]
                    det['matched'] = True
                    
                    # Re-initialize tracker with new box to correct drift
                    # Note: OpenCV trackers often require full re-init to "move" the box
                    new_tracker = create_opencv_tracker(tracker_type)
                    new_tracker.init(frame, det['bbox'])
                    trk['tracker'] = new_tracker
                    trk['bbox'] = det['bbox']
                    trk['lost_frames'] = 0
            
            # 4. Create new trackers for unmatched detections
            for det in detections:
                if not det['matched']:
                    new_tracker = create_opencv_tracker(tracker_type)
                    new_tracker.init(frame, det['bbox'])
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    active_trackers.append({
                        'tracker': new_tracker,
                        'id': next_id,
                        'bbox': det['bbox'],
                        'lost_frames': 0,
                        'color': color
                    })
                    next_id += 1
        
        # 5. Prune lost trackers
        # Remove trackers that have been lost for > 5 frames or failed update
        active_trackers = [t for t in active_trackers if t['lost_frames'] < 5]
        
        # 6. visualization and Data Collection
        for trk in active_trackers:
            x, y, w, h = map(int, trk['bbox'])
            tid = trk['id']
            color = trk['color']
            
            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {tid}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save Data
            # frame, id, x, y, w, h, conf, -1, -1, -1
            row = [frame_count + 1, tid, x, y, w, h, 1.0, -1, -1, -1]
            tracking_data.append(row)

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames_to_process} frames")
            
    end_time = time.time()
    cap.release()
    out.release()
    
    # Save Text Results
    if tracking_data:
        np.savetxt(txt_path, tracking_data, fmt='%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d', delimiter=',')
        
    # Calculate Metrics
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0
    total_unique_ids = len(set(x[1] for x in tracking_data))
    
    metrics = {
        "Video": video_name,
        "Tracker": tracker_type,
        "Inference Speed (FPS)": round(avg_fps, 2),
        "Total Unique IDs": total_unique_ids,
        "Frames Processed": frame_count
    }
    return metrics

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found.")
    
    model = RTDETR(MODEL_PATH)
    
    # OpenCV Trackers
    # Note: KCF/MOSSE do not support re-init well in some versions or are just slow to re-init.
    # CSRT is reliable but slow.
    tracker_types = ["CSRT", "KCF", "MIL"] 
    
    video_file = "Vid_9.mp4"
    
    results = []

    
    video_path = os.path.join(VIDEO_DIR, video_file)
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}")
        
    for tracker_type in tracker_types:
        try:
            metrics = process_video_opencv_with_detection(video_path, tracker_type, model, device)
            if metrics:
                results.append(metrics)
        except Exception as e:
            print(f"Error processing {video_file} with {tracker_type}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        df = pd.DataFrame(results)
        print("\n=== OpenCV+Detection Tracker Evaluation Results ===")
        print(df.to_string())
        
        csv_path = os.path.join(OUTPUT_DIR, "tracking_summary_opencv_det.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSummary results saved to {csv_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
