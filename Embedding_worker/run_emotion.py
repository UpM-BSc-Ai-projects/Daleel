# run_qdrant_ai_processing.py
# Requirements: pip install opencv-python numpy qdrant-client transformers torch torchvision pillow

import os
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, IsNullCondition
from transformers import pipeline
import torch
from helpers import extract_frame_by_count, get_person_crop_from_coords

# ===================================================================
# 1. ⚙️ --- CONFIGURATION ---
# ===================================================================

# Path to your local Qdrant database
QDRANT_DB_PATH = "./qdrant_local_db"
COLLECTION_NAME = "trial"

# Maps cam_id to video file paths
VIDEO_LOCATION_MAPPING = {
    0: r"C:\Users\themi\PycharmProjects\Capstone\Datasets\haram_vids_sample\Vid_1.mp4",
    1: r"C:\Users\themi\PycharmProjects\Capstone\Datasets\haram_vids_sample\Vid_2.mp4",
    # Add more as needed
}

# Directory to save cropped images (optional, for debugging)
CROP_SAVE_DIRECTORY = r"C:\Users\themi\PycharmProjects\Capstone\saved_crops_qdrant"

# Threshold for "Lost" emotion detection
EMOTION_THRESHOLD = 0.6  # Min confidence for "sad", "fear", or "angry" -> potentiallyLost = True

# ===================================================================
# 2. 🧠 --- LOAD AI MODEL ---
# ===================================================================

def load_emotion_model(device_num=0):
    """
    Loads the emotion classification model once.
    """
    print("Loading emotion detection model...")
    emotion_clf = pipeline(
        "image-classification", 
        model="trpakov/vit-face-expression", 
        device=device_num if device_num >= 0 and torch.cuda.is_available() else -1
    )
    print("Emotion model loaded successfully.")
    return emotion_clf

# ===================================================================
# 3. 🛠️ --- HELPER FUNCTIONS ---
# ===================================================================


def run_emotion_model(crop_image, emotion_clf):
    """
    Runs emotion detection on a cropped PIL Image.
    Returns the top prediction as a dict with 'label' and 'score'.
    """
    try:
        pred = emotion_clf(crop_image, top_k=1)
        if pred:
            return pred[0]  # {'label': '...', 'score': ...}
    except Exception as e:
        print(f"  Error in emotion model: {e}")
    
    return {"label": "unknown", "score": 0.0}

def is_potentially_lost(emotion_result, threshold=0.6):
    """
    Determines if a person is potentially lost based on emotion.
    Returns True if emotion is "sad" or "fear" with confidence above threshold.
    """
    label = emotion_result['label'].lower()
    score = emotion_result['score']
    
    if label in ["sad", "fear","angry"] and score > threshold:
        return True
    return False

# ===================================================================
# 4. 🏃 --- MAIN PROCESSING SCRIPT ---
# ===================================================================

def main():
    print("\n--- Starting Qdrant AI Processing Script ---")
    
    # Create crop save directory
    os.makedirs(CROP_SAVE_DIRECTORY, exist_ok=True)
    
    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant database at: {QDRANT_DB_PATH}")
    client = QdrantClient(path=QDRANT_DB_PATH)
    
    # 2. Load emotion model
    try:
        emotion_clf = load_emotion_model(device_num=0)
    except Exception as e:
        print(f"FATAL: Could not load emotion model: {e}")
        return
    
    # 3. Get records where potentiallyLost is None
    print(f"\nQuerying records where 'potentiallyLost' is None...")
    
    # Alternative approach: Get all records and filter in Python
    # Since IsNullCondition might have version issues, we'll filter locally
    all_records = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        limit=1000  # Adjust as needed
    )[0]
    
    # Filter for records where potentiallyLost is None
    records_to_process = [
        record for record in all_records 
        if record.payload.get('potentiallyLost') is None
    ]
    
    if not records_to_process:
        print("No records to process. Exiting.")
        return
    
    print(f"Found {len(records_to_process)} records to process.")
    
    # 4. Process each record
    for record in records_to_process:
        print(f"\n--- Processing Record ID: {record.id} ---")
        
        # Extract payload data
        payload = record.payload
        cam_id = payload['cam_id']
        coords = payload['coords']  # [x1, y1, x2, y2]
        frame_count = payload['frame_count']
        pid = payload['Pid']
        
        print(f"  Pid: {pid}, cam_id: {cam_id}, frame_count: {frame_count}, coords: {coords}")
        
        # A. Get video path
        video_path = VIDEO_LOCATION_MAPPING.get(cam_id)
        if not video_path:
            print(f"  Error: No video path found for cam_id {cam_id}. Skipping.")
            continue
        
        # B. Extract frame
        frame, frame_w, frame_h = extract_frame_by_count(video_path, frame_count)
        if frame is None:
            print("  Skipping record.")
            continue
        
        # C. Crop the person
        crop_image = get_person_crop_from_coords(frame, coords, frame_w, frame_h)
        if crop_image is None:
            print("  Skipping record.")
            continue
        
        # D. Save crop (optional, for debugging)
        try:
            save_path = os.path.join(CROP_SAVE_DIRECTORY, f"crop_id_{record.id}.jpg")
            crop_image.save(save_path)
            print(f"  Saved crop to: {save_path}")
        except Exception as e:
            print(f"  Warning: Could not save crop: {e}")
        
        # E. Run emotion model
        print("  Running emotion analysis...")
        emotion_result = run_emotion_model(crop_image, emotion_clf)
        print(f"  Emotion Result: {emotion_result}")
        
        # F. Determine potentiallyLost status
        is_lost = is_potentially_lost(emotion_result, EMOTION_THRESHOLD)
        
        # G. Update Qdrant record
        try:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"potentiallyLost": is_lost},
                points=[record.id]
            )
            print(f"  ✓ Updated potentiallyLost = {is_lost} for Record ID: {record.id}")
        except Exception as e:
            print(f"  Error updating record: {e}")
    
    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    main()