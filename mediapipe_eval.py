import cv2
import mediapipe as mp
import numpy as np
import os
from collections import OrderedDict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos"
OUTPUT_DIR = "mediapipe_results"
VIDEOS_TO_PROCESS = [f"vid_{i}.mp4" for i in range(9, 14)] 
MAX_SECONDS = 30
MODEL_PATH = 'efficientdet_lite0.tflite' # Will download auto if missing

# ==========================================
# 1. LIGHTWEIGHT TRACKER (Centroid Tracking)
# ==========================================
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance # Pixel distance threshold

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between existing objects and new centroids
            D = [] 
            for oc in object_centroids:
                row = []
                for ic in input_centroids:
                    dist = np.linalg.norm(np.array(oc) - np.array(ic))
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe model...")
        import requests
        url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
        r = requests.get(url)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Download complete.")

def run_mediapipe_tracker():
    download_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Tracker
    ct = CentroidTracker(max_disappeared=20, max_distance=100) # Tune max_distance for your resolution

    # Initialize MediaPipe Object Detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.4,
        category_allowlist=['person'] # ONLY detect people
    )
    detector = vision.ObjectDetector.create_from_options(options)

    for vid_name in VIDEOS_TO_PROCESS:
        vid_path = os.path.join(VIDEO_DIR, vid_name)
        if not os.path.exists(vid_path): continue
        
        print(f"Processing {vid_name}...")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(fps * MAX_SECONDS)

        save_path = os.path.join(OUTPUT_DIR, f"result_mp_{vid_name}")
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect
            detection_result = detector.detect(mp_image)
            
            # Extract Boxes for Tracker
            rects = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                # MediaPipe returns [x, y, width, height], we need [x1, y1, x2, y2]
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                rects.append((x1, y1, x2, y2))
            
            # Update Tracker
            objects = ct.update(rects)

            # Draw
            for (object_id, centroid) in objects.items():
                # Draw ID
                text = f"ID {object_id}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Draw Boxes (Optional)
            for (x1, y1, x2, y2) in rects:
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0: print(f"Frame {frame_count}/{max_frames}", end='\r')

        cap.release()
        out.release()
        print(f"\nFinished {vid_name}")
        
        # Reset tracker for next video
        ct = CentroidTracker(max_disappeared=20, max_distance=100)

if __name__ == "__main__":
    run_mediapipe_tracker()