import os
import pandas as pd

directory = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\person_crops_v3\yolo_coords\train"
total_frames = 0
file_counts = {}

print(f"Scanning directory: {directory}")

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        try:
            df = pd.read_csv(filepath)
            if 'frame_no' in df.columns:
                unique_frames = df['frame_no'].nunique()
                file_counts[filename] = unique_frames
                total_frames += unique_frames
                print(f"{filename}: {unique_frames} frames")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

print(f"\nTotal unique frames: {total_frames}")
