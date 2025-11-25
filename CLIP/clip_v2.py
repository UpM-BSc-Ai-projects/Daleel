import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
import os
import matplotlib.pyplot as plt


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
    
    def encode_single_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
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
            
            return features
            
        except Exception as e:
            print(f"Error processing single image: {e}")
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


def visualize_batch_results(batch_results: Dict[str, List[Tuple[str, float]]], top_k: int = 5):
    """
    Visualize top-k results from batch queries using matplotlib.
    
    Args:
        batch_results: Dictionary mapping queries to list of (image_path, score) tuples
        top_k: Number of top results to display per query
    """
    num_queries = len(batch_results)
    fig, axes = plt.subplots(num_queries, top_k, figsize=(top_k * 3, num_queries * 3))
    
    # Handle single query case
    if num_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i, (query, matches) in enumerate(batch_results.items()):
        for j, (img_path, score) in enumerate(matches[:top_k]):
            ax = axes[i, j] if num_queries > 1 else axes[0, j]
            
            # Load and display image
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with score
            title = f"{Path(img_path).name}\nScore: {score:.3f}"
            ax.set_title(title, fontsize=10)
        
        # Add query text as ylabel on first column
        if num_queries > 1:
            axes[i, 0].set_ylabel(f"'{query}'", fontsize=11, rotation=0, 
                                   ha='right', va='center', labelpad=40)
        else:
            axes[0, 0].set_ylabel(f"'{query}'", fontsize=11, rotation=0, 
                                   ha='right', va='center', labelpad=40)
    
    plt.tight_layout()
    plt.show()


# Config & Initialization of CLIP ReID
reid_system = CLIP(model_name="ViT-B/32")

gallery_dir = r"/home/mohammed/Desktop/Mohammed/UPM - Term 7/AI 491 - Capstone/Capstone.v4-full-annotated.yolov8/cropped/train+emo"
TOP_K = 100

# Collect images with multiple extensions
image_paths = []
for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]:
    image_paths.extend(list(Path(gallery_dir).glob(ext)))

if not image_paths:
    print("No images found! Please update the gallery_dir path.")
    sys.exit(1)


# # Example: Single query

# # Text description of person to find
# query = "a person with a white shirt and red pants"

# # Retrieve top matches
# results = reid_system.retrieve(
#     text_query=query,
#     image_paths=image_paths,
#     top_k=10,
#     threshold=0.25  # Optional: only return matches with similarity > 0.25
# )

# # Display results
# print(f"\nTop matches for '{query}':\n")
# for rank, (img_path, score) in enumerate(results, 1):
#     print(f"{rank}. {Path(img_path).name} - Similarity: {score:.4f}")


# Multiple queries
# queries = [
#     "a person in a wheel chair",
#     "an elderly person over 60 years old",
#     "An adult male with dark hair wearing dark navy blue shirt with dark jeans or trousers. He's also wearing a wristwatch on his left wrist.",
#     "An adult black male, with short dark hair, and wearing thin-framed rectangular glasses"
# ]
queries = [
    "An afraid person",
    "A person who is afraid",
    "An angry person",
    "A person who is angry",
    "A worried person",
    "A sad person",
    "A distressed person",
]

batch_results = reid_system.batch_retrieve(
    text_queries=queries,
    image_paths=image_paths,
    top_k=TOP_K,
    threshold=0.2
)

print("\n" + "="*60)
print("Batch query results:")
print("="*60)
for query, matches in batch_results.items():
    print(f"\nQuery: '{query}'")
    for rank, (img_path, score) in enumerate(matches, 1):
        print(f"  {rank}. {Path(img_path).name} - {score:.4f}")


# Visualize results
visualize_batch_results(batch_results, top_k=TOP_K)


