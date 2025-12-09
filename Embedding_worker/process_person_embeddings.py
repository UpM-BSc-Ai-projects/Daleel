# %%
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, IsNullCondition
import cv2
import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
# from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
import os
from transformers import pipeline
from helpers import extract_frame_by_count, get_person_crop_from_coords


# %% [markdown]
# ## Config

# %%
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Should show your GPU model
print(torch.version.cuda)  # Should display the installed CUDA version


# %%
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Path to your local Qdrant database
QDRANT_DB_PATH = "./qdrant_local_db"
COLLECTION_NAME = "trial"

# Maps cam_id to video file paths
VIDEO_LOCATION_MAPPING = {
    0: r'/home/mohammed/Desktop/Mohammed/UPM - Term 7/AI 491 - Capstone/data/Videos/Vid_1.mp4', 
    1: r'/home/mohammed/Desktop/Mohammed/UPM - Term 7/AI 491 - Capstone/data/Videos/Vid_2.mp4', 
    2: r'/home/mohammed/Desktop/Mohammed/UPM - Term 7/AI 491 - Capstone/data/Videos/Vid_3.mp4'
}

CROP_SAVE_DIRECTORY = r"C:\Users\themi\PycharmProjects\Capstone\saved_crops_qdrant"
osnet_weights = Path(r"/home/mohammed/Documents/GitHub/Tracking/osnet_x0_25_msmt17.pt")

# Dictionary mapping CLIP model names to embedding dimensions
CLIP_EMBEDDING_SIZES = {
    'RN50': 1024,
    'RN101': 512,
    'RN50x4': 640,
    'RN50x16': 768,
    'RN50x64': 1024,
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 768,
    'ViT-L/14@336px': 768
}

# Extract embedding size based on model name
CLIP_MODEL_NAME = 'ViT-B/32'  # or whatever model you're using
EMBEDDING_SIZE = CLIP_EMBEDDING_SIZES[CLIP_MODEL_NAME]

ELDERLY_THRESHOLD = 0.21
DISABLED_THRESHOLD = 0.21
EMOTION_THRESHOLD = 0.6  # Min confidence for "sad", "fear", or "angry" -> potentiallyLost = True

# %% [markdown]
# ## Reading & Accessing Existing QDrant DB

# %%
# 1. Connect to existing local storage
client = QdrantClient(path=r"/home/mohammed/Documents/GitHub/Capstone-AI-SE/Demo/qdrant_local_db", force_disable_check_same_thread=True)

# %%
# 2. Verifying local qdrant local database
records = client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True, with_vectors=False)[0]
print(f"\nTotal records in database: {len(records)}")

# %%
# Show first few records
print("\nFirst 3 records:")
for r in records[:3]:
    print(f"  ID: {r.id}, Payload: {r.payload}")

# Show summary by Pid and cam_id
from collections import Counter
pids = [r.payload['Pid'] for r in records]
cam_ids = [r.payload['cam_id'] for r in records]
print(f"\nSummary:")
print(f"  PIDs: {dict(Counter(pids))}")
print(f"  Camera IDs: {dict(Counter(cam_ids))}")

# %% [markdown]
# ## CLIP

# %%
class CLIP:
    """
    CLIP-based Person Re-Identification system.
    Uses text descriptions to retrieve matching persons from an image gallery.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model for ReID.
        
        Args:
            model_name: CLIP model variant. Options: "RN50", "RN101", "RN50x4", 
                       "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"
            device: Device to run model on. If None, auto-detects GPU/CPU.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
    
    def encode_images(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """
        Encode a list of images into feature vectors.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Normalized image feature tensor of shape (N, feature_dim)
        """
        image_features = []
        
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model.encode_image(image_input)
                    features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                    
                image_features.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not image_features:
            raise ValueError("No images were successfully processed")
            
        return torch.cat(image_features, dim=0)
    
    def encode_person_crop(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode a single person image into a feature vector.
        
        Args:
            image: Input image (PIL cropped person image)
            
        Returns:
            Normalized image feature tensor of shape (1, feature_dim)
        """
        try:
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                features = features.squeeze(0)
            
            return features
            
        except Exception as e:
            raise ValueError(f"Failed to encode single image: {e}")
    
    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        """
        Encode text descriptions into feature vectors.
        
        Args:
            text_descriptions: List of text descriptions
            
        Returns:
            Normalized text feature tensor of shape (N, feature_dim)
        """
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def retrieve(self, 
                 text_query: str,
                 image_paths: List[Union[str, Path]],
                 top_k: int = 10,
                 threshold: float = None) -> List[Tuple[str, float]]:
        """
        Retrieve images matching the text description.
        
        Args:
            text_query: Text description of the person to find
            image_paths: List of paths to gallery images
            top_k: Number of top matches to return
            threshold: Optional similarity threshold (0-1). Only return matches above this.
            
        Returns:
            List of tuples (image_path, similarity_score) sorted by similarity
        """
        print(f"Encoding {len(image_paths)} images...")
        image_features = self.encode_images(image_paths)
        
        print(f"Encoding text query: '{text_query}'")
        text_features = self.encode_text([text_query])
        
        # Compute cosine similarity
        similarities = (image_features @ text_features.T).squeeze().cpu().numpy()
        
        # Handle single image case
        if similarities.ndim == 0:
            similarities = np.array([similarities])
        
        # Get top-k indices
        top_k = min(top_k, len(image_paths))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if threshold is None or score >= threshold:
                results.append((str(image_paths[idx]), score))
        
        return results
    
    def batch_retrieve(self,
                      text_queries: List[str],
                      image_paths: List[Union[str, Path]],
                      top_k: int = 10,
                      threshold: float = None) -> dict:
        """
        Retrieve images for multiple text queries at once.
        
        Args:
            text_queries: List of text descriptions
            image_paths: List of paths to gallery images
            top_k: Number of top matches to return per query
            threshold: Optional similarity threshold (0-1). Only return matches above this.
            
        Returns:
            Dictionary mapping each query to its list of (image_path, score) tuples
        """
        print(f"Encoding {len(image_paths)} images...")
        image_features = self.encode_images(image_paths)
        
        print(f"Encoding {len(text_queries)} text queries...")
        text_features = self.encode_text(text_queries)
        
        # Compute all similarities at once
        similarities = (image_features @ text_features.T).cpu().numpy()  # Shape: (n_images, n_queries)
        
        results = {}
        for i, query in enumerate(text_queries):
            query_sims = similarities[:, i]
            top_k_actual = min(top_k, len(image_paths))
            top_indices = np.argsort(query_sims)[::-1][:top_k_actual]
            
            results[query] = [
                (str(image_paths[idx]), float(query_sims[idx]))
                for idx in top_indices
                if (threshold is None or float(query_sims[idx]) >= threshold)
            ]
        
        return results

# %%
REID_SYSTEM = CLIP(model_name=CLIP_MODEL_NAME)

# %% [markdown]
# ## Load emotion classification model

# %%
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

# %% [markdown]
# ### Functions for determing emotion & lost status

# %%
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

# %%

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

# %% [markdown]
# ## Creating a Qdrant collection (table) for the text embeddings

# %%
client.delete_collection(collection_name='text_embeddings')
client.create_collection(
    collection_name='text_embeddings',
    vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
)

texts = [
    "an elderly person over 60 years old",
    "a person in a wheel chair"
]

prompt_embeddings = REID_SYSTEM.encode_text(texts).to('cpu')

elderly_person_embedding = prompt_embeddings[0].numpy()
disabled_person_embedding = prompt_embeddings[1].numpy()

print(prompt_embeddings.shape)

points = [
    PointStruct(id=1, vector=elderly_person_embedding, payload={'title': 'elderly person prompt embedding'}),
    PointStruct(id=2, vector=disabled_person_embedding, payload={'title': 'disabled person prompt embedding'})
]

client.upsert(collection_name='text_embeddings', points=points)

# %% [markdown]
# ### Functions to determine if a person is elderly or disabled

# %%
def determine_elderly(person_emb, threshold=0.6):
    # Retrieve specific record by ID
    elderly_result = client.retrieve(
        collection_name="text_embeddings",
        ids=[1], # id 1 is the elderly person embedding
        with_vectors=True
    )

    # Extract the vector and payload
    if elderly_result:
        elderly_point = elderly_result[0]
        elderly_embedding = elderly_point.vector
        elderly_payload = elderly_point.payload
        print(f"ID: {elderly_point.id}")
        print(f"Vector: {elderly_embedding}")
        print(f"Payload: {elderly_payload}")
    else:
        print("No elderly person embedding found.")
        return False
    
    cos_sim = np.dot(person_emb, elderly_embedding) # Because it's already normalized, cosine similarity is the dot product
    print(f"Cosine Similarity: {cos_sim}")
    
    if cos_sim > threshold:
        return True
    return False
        

# %%
def determine_disabled(person_emb, threshold=0.6):
    # Retrieve specific record by ID
    disabled_result = client.retrieve(
        collection_name="text_embeddings",
        ids=[2], # id 2 is the disabled person embedding
        with_vectors=True
    )

    # Extract the vector and payload
    if disabled_result:
        disabled_point = disabled_result[0]
        disabled_embedding = disabled_point.vector
        disabled_payload = disabled_point.payload
        print(f"ID: {disabled_point.id}")
        print(f"Vector: {disabled_embedding}")
        print(f"Payload: {disabled_payload}")
    else:
        print("No disabled person embedding found.")
        return False
    
    cos_sim = np.dot(person_emb, disabled_embedding) # Because it's already normalized, cosine similarity is the dot product
    print(f"Cosine Similarity: {cos_sim}")
    
    if cos_sim > threshold:
        return True
    return False


# %%
prompt_records = client.scroll(collection_name='text_embeddings', limit=100, with_payload=True, with_vectors=True)[0]
print(f"\nTotal records in database: {len(prompt_records)}")

# Show first few records
print(f"\nRecords:")
for r in prompt_records[:3]:
    print(f"  ID: {r.id}, Payload: {r.payload}")

# %% [markdown]
# ## Creating a Qdrant collection for CLIP & OSnet embeddings

# %%
client.delete_collection(collection_name='CLIP_embeddings')
client.delete_collection(collection_name='osnet_embeddings')

client.create_collection(
    collection_name='CLIP_embeddings',
    vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
)

client.create_collection(
    collection_name='osnet_embeddings',
    vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
)

# %% [markdown]
# ### Create OSNet embedding

# %%
def get_osnet_embedding(pil_image):
    # Convert PIL to numpy array (RGB format, which PIL uses by default)
    image_np = np.array(pil_image)

    # Initialize model
    model = ReidAutoBackend(
        weights=osnet_weights,
        device=DEVICE,
        half=False
    )

    # Get the actual model and move to device
    osnet_model = model.model.model.to(DEVICE)
    
    # Convert image to tensor and preprocess
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess image
    image_tensor = transform(image_np).unsqueeze(0).to(DEVICE)
    
    # Get embedding
    with torch.no_grad():
        osnet_embedding = osnet_model(image_tensor).cpu().numpy()[0]

    # Normalize
    osnet_embedding = osnet_embedding / np.linalg.norm(osnet_embedding)

    print(f"OSNet Embedding shape: {osnet_embedding.shape}")
    return osnet_embedding

# %% [markdown]
# ## MAIN

# %%
def main():
    # Create crop save directory
    os.makedirs(CROP_SAVE_DIRECTORY, exist_ok=True)
    
    # 2. Load emotion model
    try:
        emotion_clf = load_emotion_model(device_num=0)
    except Exception as e:
        print(f"FATAL: Could not load emotion model: {e}")
        return
    
    # 3. Get records where potentiallyLost is None
    print(f"\nQuerying records where 'potentiallyLost' is None...")
    
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
    
    print(f"Found {len(records_to_process)} records to process.")
    
    if not records_to_process:
        print("No records to process. Exiting.")
        return
    
    # 4. Process each record
    for record in records_to_process:
        print(f"\n--- Processing Record ID: {record.id} ---")
        
        # Extract payload data
        payload = record.payload
        cam_id = payload['cam_id']
        coords = payload['coords']  # [x1, y1, x2, y2]
        frame_count = payload['frame_count']
        pid = payload['Pid']
        
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
        
        # G. Create an embedding for the person
        clip_person_emb = REID_SYSTEM.encode_person_crop(crop_image).to('cpu').numpy()
        print(clip_person_emb.shape)
        
        osnet_person_emb = get_osnet_embedding(crop_image)
        
        # H. Determine Elderly status
        is_elderly = determine_elderly(clip_person_emb, ELDERLY_THRESHOLD)
        
        # I. Determine Disabled status
        is_disabled = determine_disabled(clip_person_emb, DISABLED_THRESHOLD)
        
        
        # J. Update Qdrant record
        try:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"isElderly": is_elderly, "isDisabled": is_disabled, "potentiallyLost": is_lost},
                points=[record.id]
            )
            print(f"  ✓ Updated isElderly = {is_elderly} and isDisabled = {is_disabled} and potentiallyLost = {is_lost} for Record ID: {record.id}")
        except Exception as e:
            print(f"  Error updating record: {e}")
            
        # K. Add record to CLIP & OSNet collections
        clip_points = [PointStruct(id=record.id, vector=clip_person_emb, payload={"Pid": pid})]
        osnet_points = [PointStruct(id=record.id, vector=osnet_person_emb, payload={"Pid": pid})]
        
        client.upsert(collection_name='CLIP_embeddings', points=clip_points)
        client.upsert(collection_name='osnet_embeddings', points=osnet_points)
        
    print("\n--- Processing Complete ---")

# %%
if __name__ == "__main__":
    main()

# %%



