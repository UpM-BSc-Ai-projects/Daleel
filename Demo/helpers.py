import cv2
import os
from PIL import Image

def extract_frame_by_count(video_path, frame_count):
    """
    Opens a video and extracts the frame at the given frame_count.
    Returns the frame and its dimensions (width, height).
    """
    if not os.path.exists(video_path):
        print(f"  Error: Video file not found: {video_path}")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file: {video_path}")
        return None, None, None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"  Successfully extracted frame at count {frame_count}.")
        return frame, frame_width, frame_height
    else:
        print(f"  Error: Could not read frame {frame_count} from {video_path}.")
        return None, None, None

def get_person_crop_from_coords(frame, coords, frame_w, frame_h):
    """
    Crops a person from an OpenCV frame using absolute pixel coordinates.
    coords format: [x1, y1, x2, y2]
    Returns a PIL Image.
    """
    try:
        x1, y1, x2, y2 = map(int, coords)
        
        # Clamp values to be within the frame dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)
        
        if x1 >= x2 or y1 >= y2:
            print(f"  Error: Invalid coordinates after clamping. Box: [{x1}, {y1}, {x2}, {y2}]")
            return None

    except Exception as e:
        print(f"  Error: Invalid coordinates format {coords}. {e}")
        return None

    # Crop the frame
    cropped_frame = frame[y1:y2, x1:x2]
    
    if cropped_frame.size == 0:
        print(f"  Error: Crop size is zero. Box: [{x1}, {y1}, {x2}, {y2}]")
        return None
    
    # Convert from OpenCV BGR to PIL RGB
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return pil_image