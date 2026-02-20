import os
import cv2
import pandas as pd
import random
from tqdm import tqdm

# Paths
VIDEO_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos\train"
CSV_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\person_crops_v3\yolo_coords\train"
DATA_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Detection Fine-tune\data"

# Create directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(DATA_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'labels', split), exist_ok=True)

def process_video(video_filename, csv_filename):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    csv_path = os.path.join(CSV_DIR, csv_filename)
    
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        print(f"Skipping {video_filename} (missing files)")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_filename}: {e}")
        return []

    grouped = df.groupby('frame_no')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open {video_filename}")
        return []

    annotated_frames = sorted(df['frame_no'].unique())
    samples = []

    for frame_no in tqdm(annotated_frames, desc=video_filename, leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret: continue

        base_name = f"{os.path.splitext(video_filename)[0]}_frame_{frame_no}"
        
        # Prepare label data
        labels = []
        frame_data = grouped.get_group(frame_no)
        for _, row in frame_data.iterrows():
            # Class 0 (person), xywh normalized
            xc, yc = max(0, min(1, row['x_center'])), max(0, min(1, row['y_center']))
            w, h = max(0, min(1, row['width'])), max(0, min(1, row['height']))
            labels.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        
        samples.append({
            'base_name': base_name,
            'image': frame,
            'labels': "\n".join(labels)
        })

    cap.release()
    return samples

# Process all files
all_samples = []
csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]

print(f"Processing {len(csv_files)} videos...")
for csv_file in csv_files:
    video_file = csv_file.replace('.csv', '.mp4')
    all_samples.extend(process_video(video_file, csv_file))

# Shuffle and Split (80/20)
random.seed(42)
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.8)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

def save_split(samples, split):
    print(f"Saving {len(samples)} images to {split}...")
    for s in samples:
        cv2.imwrite(os.path.join(DATA_DIR, 'images', split, s['base_name'] + '.jpg'), s['image'])
        with open(os.path.join(DATA_DIR, 'labels', split, s['base_name'] + '.txt'), 'w') as f:
            f.write(s['labels'])

save_split(train_samples, 'train')
save_split(val_samples, 'val')
print(f"Done! Data saved to {DATA_DIR}")
