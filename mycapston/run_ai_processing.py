# run_ai_processing.py
# Requirements: pip install opencv-python numpy
import os
import django
import sys
import cv2  # OpenCV for video processing
import numpy as np
from PIL import Image
from datetime import datetime

# ===================================================================
# 1. ⚙️ --- CONFIGURATION (Your Updates) ---
# ===================================================================

# Full path to your trained .pth model
YOUR_SPECIAL_MODEL_PATH = r"C:\Users\themi\PycharmProjects\Capstone\Datasets\Disabled\Models\mobilenet_best_R2.pth"

# Maps 'location' string from LastSeen to a video file
VIDEO_LOCATION_MAPPING = {
    "Camera V1": r"C:\Users\themi\PycharmProjects\Capstone\Datasets\haram_vids_sample\Vid_1.mp4",
    "Camera V2": r"C:\Users\themi\PycharmProjects\Capstone\Datasets\haram_vids_sample\Vid_2.mp4",
    "Camera V3": r"C:\Users\themi\PycharmProjects\Capstone\Datasets\haram_vids_sample\Vid_3.mp4",
}

# --- NEW: Directory to save the cropped images ---
CROP_SAVE_DIRECTORY = r"C:\Users\themi\PycharmProjects\Capstone\saved_crops"
# --------------------------------------------------

# Adjustable thresholds for a "True" status
AI_THRESHOLDS = {
    "emotion": 0.6,  # Min confidence for "Lost"
    "special": 0.7,  # Min confidence for "special" -> isDisabled
    "age": 0.1       # Min confidence for an age to be considered
}

# ===================================================================
# 2. 🚀 --- DJANGO AND AI SCRIPT SETUP (Your Updates) ---
# ===================================================================

print("Setting up Django environment...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', f"mycapston.settings")
try:
    django.setup()
except ModuleNotFoundError:
    print(f"Error: Could not find Django settings 'mycapston.settings'")
    print("Please make sure 'mycapston.settings' is set correctly.")
    sys.exit(1)

print("Importing models and AI pipeline...")
try:
    # Import your models *after* setup
    from base.models import LastSeen, CameraDetectedPerson
except ImportError:
    print(f"Error: Could not import models from 'base'.")
    print("Please make sure 'base' is set correctly.")
    sys.exit(1)

try:
    from ai_pipeline import load_all_pipelines, run_ai_on_crop
except ImportError:
    print("Error: Could not import from 'ai_pipeline.py'.")
    print("Make sure 'ai_pipeline.py' is in the same directory.")
    sys.exit(1)

# ===================================================================
# 3. 🧠 --- HELPER FUNCTIONS (MERGED & CORRECTED) ---
# ===================================================================
# (All helper functions remain the same as your provided version)

def parse_time_string(time_str):
    """
    Parses a time string like "HH:MM:SS:MS" or "HH:MM:SS"
    into total milliseconds.
    Assumes the last part is milliseconds if 4 parts are present.
    """
    try:
        parts = str(time_str).split(':')
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
            total_ms = (h * 3600 + m * 60 + s) * 1000 + ms
            return total_ms
        elif len(parts) == 3: # Handle "HH:MM:SS"
            h, m, s = map(int, parts)
            total_ms = (h * 3600 + m * 60 + s) * 1000
            return total_ms
        else:
            print(f"  Error: Unrecognized time format: {time_str}")
            return None
    except Exception as e:
        print(f"  Error parsing time string '{time_str}': {e}")
        return None

def extract_frame(video_path, time_str):
    """
    Opens a video, seeks to the exact time string, and returns a single frame
    along with the frame's width and height.
    """
    if not os.path.exists(video_path):
        print(f"  Error: Video file not found: {video_path}")
        return None, None, None

    time_ms = parse_time_string(time_str)
    if time_ms is None:
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file: {video_path}")
        return None, None, None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"  Successfully extracted frame at {time_ms}ms.")
        return frame, frame_width, frame_height
    else:
        print(f"  Error: Could not read frame at {time_ms}ms from {video_path}.")
        return None, None, None

def get_person_crop_from_yolo(frame, yolo_coords, frame_w, frame_h):
    """
    Crops a person from an OpenCV frame using
    normalized YOLO [xc, yc, w, h] coordinates.
    Returns a PIL Image.
    """
    try:
        # Denormalize YOLO coordinates
        xc, yc, w, h = map(float, yolo_coords)
        cx = xc * frame_w
        cy = yc * frame_h
        bw = w * frame_w
        bh = h * frame_h
        
        # Convert to [x1, y1, x2, y2]
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))

        # Clamp values to be within the frame dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)
        
        if x1 >= x2 or y1 >= y2:
            print(f"  Error: Invalid coordinates after clamping (width/height is zero). Box: [{x1}, {y1}, {x2}, {y2}]")
            return None

    except Exception as e:
        print(f"  Error: Invalid coordinates format {yolo_coords}. {e}")
        return None

    # Crop the frame (which is a NumPy array)
    cropped_frame = frame[y1:y2, x1:x2]
    
    if cropped_frame.size == 0:
        print(f"  Error: Crop size is zero. Original box: [{x1}, {y1}, {x2}, {y2}]")
        return None
    
    # Convert from OpenCV BGR to PIL RGB
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return pil_image

def is_age_elderly(age_label):
    """
    Checks if an age label is "elderly" (50+).
    !!! Review this to match your age model's output labels !!!
    """
    age_label = str(age_label).lower()
    # Example labels from 'nateraw/vit-age-classifier'
    elderly_labels = ['50-59', '60-69', '70+', 'more than 70'] 
    
    for label in elderly_labels:
        if label in age_label:
            return True
    return False

# ===================================================================
# 4. 🏁 --- MAIN PROCESSING SCRIPT (WITH YOUR 2 ADDITIONS) ---
# ===================================================================

def main():
    print("\n--- Starting AI Processing Script ---")
    
    # --- NEW: Ensure the crop save directory exists ---
    os.makedirs(CROP_SAVE_DIRECTORY, exist_ok=True)
    # -------------------------------------------------
    
    # 1. Load AI Models (only once)
    try:
        models = load_all_pipelines(YOUR_SPECIAL_MODEL_PATH, device_num=0)
    except Exception as e:
        print(f"FATAL: Could not load AI models: {e}")
        return

    # 2. Get records from Database
    records_to_process = LastSeen.objects.filter(
        CDPid__ai_processed=False
    ).select_related('CDPid')
    
    if not records_to_process.exists():
        print("No new records to process. Exiting.")
        return
        
    print(f"Found {len(records_to_process)} records to process.")
    
    # --- NEW: Set to track CDP IDs processed in this run ---
    processed_cdp_ids_in_this_run = set()
    # -------------------------------------------------------

    # 3. Loop and process each record
    for record in records_to_process:
        print(f"\nProcessing LastSeen ID: {record.id} (for CDP ID: {record.CDPid.cameraDetectedPersonId})")
        
        cdp = record.CDPid # The CameraDetectedPerson object to update
        
        # --- NEW: Check if this CDP was already processed in this batch ---
        if cdp.cameraDetectedPersonId in processed_cdp_ids_in_this_run:
            print(f"  Info: CDP ID {cdp.cameraDetectedPersonId} was already processed in this run. Skipping LastSeen ID {record.id}.")
            continue
        # ------------------------------------------------------------------
        
        # A. Get video path from mapping
        video_path = VIDEO_LOCATION_MAPPING.get(record.location)
        if not video_path:
            print(f"  Error: No video path found for location '{record.location}'. Skipping.")
            continue
            
        # B. Extract the frame from the video (NOW returns frame + dimensions)
        frame, frame_w, frame_h = extract_frame(video_path, record.time)
        if frame is None:
            print("  Skipping record.")
            continue
            
        # C. Crop the person from the frame (NOW uses YOLO coords)
        crop_image = get_person_crop_from_yolo(frame, record.coordinates, frame_w, frame_h)
        if crop_image is None:
            print("  Skipping record.")
            continue
            
        # --- NEW: Save the crop to a file ---
        try:
            # We use the LastSeen ID for a unique filename
            save_path = os.path.join(CROP_SAVE_DIRECTORY, f"crop_{record.id}.jpg")
            crop_image.save(save_path)
            print(f"  Successfully saved crop to {save_path}")
        except Exception as e:
            print(f"  Warning: Could not save crop image: {e}")
        # ------------------------------------
            
        # D. Run AI models on the crop
        print("  Running AI analysis...")
        ai_results = run_ai_on_crop(crop_image, models)
        print(f"  AI Results: {ai_results}")
        
        # E. Update the CameraDetectedPerson (CDP) object
        
        # Update potentiallyLost
        emotion_res = ai_results['emotion']
        if emotion_res['label'].lower() in ["sad", "fear"] and emotion_res['score'] > AI_THRESHOLDS['emotion']:
            cdp.potentiallyLost = True
            print("  Status: Set potentiallyLost = True")

        # Update isDisabled
        special_res = ai_results['special']
        if special_res['label'] == "special" and special_res['score'] > AI_THRESHOLDS['special']:
            cdp.isDisabled = True
            print("  Status: Set isDisabled = True")
            
        # Update isElderly
        age_res = ai_results['age']
        if is_age_elderly(age_res['label']) and age_res['score'] > AI_THRESHOLDS['age']:
            cdp.isElderly = True
            print(f"  Status: Set isElderly = True (Age: {age_res['label']})")
            
        # F. Mark as processed and save
        cdp.ai_processed = True
        cdp.save(update_fields=['potentiallyLost', 'isDisabled', 'isElderly', 'ai_processed'])
        print(f"  Successfully processed and saved CDP ID: {cdp.cameraDetectedPersonId}")

        # --- NEW: Mark this CDP as processed for this run ---
        processed_cdp_ids_in_this_run.add(cdp.cameraDetectedPersonId)
        # ----------------------------------------------------

    print("\n--- Processing complete. ---")


if __name__ == "__main__":
    main()