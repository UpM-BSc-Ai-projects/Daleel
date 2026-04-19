import os
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import warnings
from io import BytesIO
from PIL import Image

# Third-party models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\person_crops_v3\crops"

print("Loading CLIP model into memory (This might take a moment if it needs to download weights)...")
clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device="cuda" if torch.cuda.is_available() else "cpu")

def get_images(names=None, count=3):
    image_paths = []
    if names:
        print(f"Searching for specified images: {names} ...")
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                if file in names:
                    image_paths.append(os.path.join(root, file))
    else:
        print(f"Selecting {count} random images from {DATASET_DIR} ...")
        all_images = []
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    all_images.append(os.path.join(root, file))
        if len(all_images) >= count:
            image_paths = random.sample(all_images, count)
        else:
            image_paths = all_images
            
    return image_paths

def compute_blur_variance(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def process_optimization(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    orig_size = os.path.getsize(img_path)
    
    # 4. Absolute Quality Gating: Laplacian variance
    variance = compute_blur_variance(img)
    is_blurry = variance < 100.0  # Threshold can be tuned based on dataset
    
    results = [
        {"name": f"Original\nBlurVar: {variance:.1f}", "img": img.copy(), "size": orig_size, "type": "Base"}
    ]
    
    if is_blurry:
        results[0]["name"] = f"Original (DROPPED!)\nBlurVar: {variance:.1f} < 100"
    
    # 1. Bounding Box Shrinking (Tightening the Crop) - Cancelled by User
    
    # 2. Extreme Lossy Tuning (WebP Q=80, 60, 40, 20)
    for q in [80, 60, 40, 20]:
        b = BytesIO()
        img.save(b, format='WEBP', quality=q)
        size = b.getbuffer().nbytes
        b.seek(0)
        results.append({"name": f"2. WebP (Q={q})", "img": Image.open(b).copy(), "size": size, "type": "Extreme Lossy"})
        
    # 3. Downsampling
    for scale in [90, 80]:
        w, h = img.size
        new_w = int(w * (scale / 100.0))
        new_h = int(h * (scale / 100.0))
        if new_w > 0 and new_h > 0:
            downscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            downscaled = img.copy()
            
        b = BytesIO()
        downscaled.save(b, format='WEBP', quality=40)
        size = b.getbuffer().nbytes
        b.seek(0)
        results.append({"name": f"3. Scale {scale}%\nWebP(Q=60)", "img": Image.open(b).copy(), "size": size, "type": "Scaled"})

    # Evaluate CLIP similarity against original
    print(f"Evaluating Semantic CLIP loss for {len(results)} variants...")
    images_for_clip = [res["img"].convert("RGB") for res in results]
    vectors = clip_model.encode(images_for_clip)
    
    orig_vector = vectors[0].reshape(1, -1)
    for i, res in enumerate(results):
        if i == 0:
            res["similarity"] = 1.0
        else:
            sim = cosine_similarity(orig_vector, vectors[i].reshape(1, -1))[0][0]
            res["similarity"] = float(sim)

    return results

def plot_comparisons(image_paths):
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found.")
        return
        
    # We will generate results once for sample to get the tech count
    sample_results = process_optimization(image_paths[0])
    num_techniques = len(sample_results)
    
    fig, axes = plt.subplots(num_images, num_techniques, figsize=(2.5 * num_techniques, 4 * num_images))
    if num_images == 1:
        axes = [axes]
        
    for i, path in enumerate(image_paths):
        print(f"Processing image pipeline {i+1}/{num_images}...")
        results = sample_results if i == 0 else process_optimization(path)
            
        for j, res in enumerate(results):
            ax = axes[i][j]
            ax.imshow(res["img"])
            ax.axis('off')
            
            size_kb = res["size"] / 1024.0
            
            if "Original" in res["name"]:
                color = "black"
                if "DROPPED" in res["name"]:
                    color = "red"
            else:
                orig_kb = results[0]["size"] / 1024.0
                ratio = (size_kb / orig_kb) if orig_kb > 0 else 0
                color = "green" if ratio <= 1.0 else "red"
                
            title = f'{res["name"]}\nSize: {size_kb:.1f} KB\nCLIP Sim: {res["similarity"]*100:.1f}%'
            ax.set_title(title, fontsize=9, color=color)
            
            # Print text summary
            name_clean = res["name"].replace('\n', ' ')
            print(f"Image {i+1} | {name_clean} | Size: {size_kb:.1f} KB | CLIP: {res['similarity']*100:.1f}%")
            
    plt.tight_layout()
    plt.show()
    plt.savefig("optimization_results.png")
    print("Saved plot to optimization_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize experimental optimization methods")
    parser.add_argument('--names', nargs='+', help="Provide specific image filenames (e.g., 0001.jpg 0002.jpg). Random if unused.")
    args = parser.parse_args()
    
    images = get_images(args.names, count=3)
    plot_comparisons(images)
