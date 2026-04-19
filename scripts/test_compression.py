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
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\person_crops_v3\crops"

print("Loading CLIP and YOLO models into memory (This might take a moment if it needs to download weights)...")
clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device="cuda" if torch.cuda.is_available() else "cpu")
yolo8 = YOLO("yolov8n-seg.pt")
yolo11 = YOLO("yolo11n-seg.pt")

def remove_background(img, model):
    res = model(img, verbose=False)
    if not res or res[0].masks is None:
        return img.copy().convert("RGBA")
        
    # Find person class (0 in COCO)
    classes = res[0].boxes.cls
    person_idx = torch.where(classes == 0)[0]
    
    if len(person_idx) == 0:
        return img.copy().convert("RGBA")
        
    masks = res[0].masks.data[person_idx]
    # Union of all person masks
    mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (img.width, img.height), interpolation=cv2.INTER_NEAREST)
    
    img_rgba = img.convert("RGBA")
    np_img = np.array(img_rgba)
    np_img[:, :, 3] = mask_resized
    return Image.fromarray(np_img)

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

def compress_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    orig_size = os.path.getsize(img_path)
    results = [
        {"name": "Original", "img": img.copy(), "size": orig_size, "type": "Base"}
    ]
    

    # 2. PIL WebP 80 (Lossy)
    b = BytesIO()
    img.save(b, format='WEBP', quality=80)
    size = b.getbuffer().nbytes
    b.seek(0)
    results.append({"name": "PIL WebP (Q=80)", "img": Image.open(b).copy(), "size": size, "type": "Lossy"})

    # Convert to CV2 Matrix (BGR format) for OpenCV processing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


    # 5. Try scikit-image TV denoising
    try:
        from skimage.restoration import denoise_tv_chambolle
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = denoise_tv_chambolle(np.array(img), weight=0.1, channel_axis=-1)
            denoised_uint8 = (denoised * 255).astype(np.uint8)
            
            b = BytesIO()
            denoised_img = Image.fromarray(denoised_uint8)
            denoised_img.save(b, format='WEBP', quality=80)
            size = b.getbuffer().nbytes
            b.seek(0)
            
            results.append({"name": "skimage TV Denoise -> WEBP(80)", "img": Image.open(b).copy(), "size": size, "type": "Lossy Filter"})
            
    except ImportError:
        pass

    # 6. YOLOv8 Segmentation BG Removal -> WebP
    bg_removed_8 = remove_background(img, yolo8)
    b = BytesIO()
    bg_removed_8.save(b, format='WEBP', quality=80) 
    size = b.getbuffer().nbytes
    b.seek(0)
    results.append({"name": "YOLOv8 Seg BG-Rm -> WebP(80)", "img": Image.open(b).copy(), "size": size, "type": "Object ISO"})

    # 7. YOLO11 Segmentation BG Removal -> WebP
    bg_removed_11 = remove_background(img, yolo11)
    b = BytesIO()
    bg_removed_11.save(b, format='WEBP', quality=80) 
    size = b.getbuffer().nbytes
    b.seek(0)
    results.append({"name": "YOLO11 Seg BG-Rm -> WebP(80)", "img": Image.open(b).copy(), "size": size, "type": "Object ISO"})

    # Evaluate CLIP similarity
    print(f"Evaluating Semantic CLIP loss for {len(results)} variants...")
    images_for_clip = [res["img"].convert("RGB") for res in results]
    vectors = clip_model.encode(images_for_clip)
    
    orig_vector = vectors[0].reshape(1, -1)
    for i, res in enumerate(results):
        sim = cosine_similarity(orig_vector, vectors[i].reshape(1, -1))[0][0]
        res["similarity"] = float(sim)

    return results

def plot_comparisons(image_paths):
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found.")
        return
        
    sample_results = compress_image(image_paths[0])
    num_techniques = len(sample_results)
    
    # Create matplotlib grid
    fig, axes = plt.subplots(num_images, num_techniques, figsize=(3 * num_techniques, 4 * num_images))
    if num_images == 1:
        axes = [axes]
        
    for i, path in enumerate(image_paths):
        print(f"Processing image pipeline {i+1}/{num_images}...")
        # Since it's the first image again, don't run it twice. We can just run compress once for all
        if i == 0:
            results = sample_results
        else:
            results = compress_image(path)
            
        for j, res in enumerate(results):
            ax = axes[i][j]
            ax.imshow(res["img"])
            ax.axis('off')
            
            size_kb = res["size"] / 1024.0
            
            if res["name"] == "Original":
                color = "black"
            else:
                orig_kb = results[0]["size"] / 1024.0
                ratio = (size_kb / orig_kb) if orig_kb > 0 else 0
                color = "green" if ratio <= 1.0 else "red"
                
            title = f'{res["name"]}\nSize: {size_kb:.1f} KB\nCLIP Sim: {res["similarity"]*100:.1f}%'
            ax.set_title(title, fontsize=9, color=color)
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize different image compression methods")
    parser.add_argument('--names', nargs='+', help="Provide specific image filenames (e.g., 0001.jpg 0002.jpg). Random if unused.")
    args = parser.parse_args()
    
    images = get_images(args.names, count=3)
    plot_comparisons(images)
