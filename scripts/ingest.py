import os
from dotenv import load_dotenv
import re
import uuid
import logging
import torch
from io import BytesIO
from PIL import Image
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
DATASET_DIR = os.getenv("DATASET_DIR", r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\v10")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "image-dataset")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "images")
MODEL_NAME = os.getenv("CLIP_MODEL", "sentence-transformers/clip-ViT-B-32")

def initialize_clients():
    """Initializes MinIO and Qdrant clients, creating buckets and collections if they don't exist."""
    logger.info("Initializing clients...")
    
    # MinIO Client
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        logger.info(f"Created MinIO bucket: {BUCKET_NAME}")
    else:
        logger.info(f"MinIO bucket {BUCKET_NAME} already exists.")

    # Qdrant Client
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Initialize Model to get vector size
    logger.info(f"Loading model {MODEL_NAME}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        logger.info("get_sentence_embedding_dimension() returned None. Using dummy encode to find size.")
        vector_size = len(model.encode("dummy text").tolist())
    
    collections = qdrant_client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME} with vector size {vector_size}")
    else:
        logger.info(f"Qdrant collection {COLLECTION_NAME} already exists.")
        
    return minio_client, qdrant_client, model

BATCH_SIZE = 500

def ingest_dataset():
    """Main ingestion loop to process all images in the dataset directory using batches for CUDA."""
    minio_client, qdrant_client, model = initialize_clients()
    
    logger.info(f"Starting ingestion from {DATASET_DIR}...")
    
    # Collect all image paths
    image_paths = []
    target_vids = {'Vid_11','Vid_12','Vid_13','Vid_14'}
    
    for root, _, files in os.walk(DATASET_DIR):
        # We only want to process images if they are inside one of our target video directories
        path_parts = set(root.split(os.sep))
        if not target_vids.intersection(path_parts):
            continue
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
                
    total_images = len(image_paths)
    logger.info(f"Found {total_images} images to process.")
    
    success_count = 0
    fail_count = 0
    
    # Process in batches
    for i in range(0, total_images, BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1}/{(total_images + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_paths)} images)...")
        
        batch_images = []
        batch_ids = []
        batch_byte_arrs = []
        batch_sizes = []
        valid_paths = []
        
        # Load and compress images
        for path in batch_paths:
            try:
                base_name = os.path.splitext(os.path.basename(path))[0]
                image_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, base_name))
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Scale down to 90%
                    w, h = img.size
                    new_w = int(w * 0.9)
                    new_h = int(h * 0.9)
                    if new_w > 0 and new_h > 0:
                        img_copy = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        img_copy = img.copy()
                    
                    img_byte_arr = BytesIO()
                    img_copy.save(img_byte_arr, format='WEBP', quality=60)
                    img_byte_arr.seek(0)
                    img_size = img_byte_arr.getbuffer().nbytes
                    
                    batch_images.append(img.copy())
                    batch_ids.append(image_id)
                    batch_byte_arrs.append(img_byte_arr)
                    batch_sizes.append(img_size)
                    valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                fail_count += 1
                
        if not batch_images:
            continue
            
        # Vectorize batch using PyTorch CUDA dynamically within SentenceTransformer
        try:
            vectors = model.encode(batch_images, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()
        except Exception as e:
            logger.error(f"Error vectorizing batch: {e}")
            fail_count += len(batch_images)
            continue
            
        # Upload to MinIO and prepare Qdrant points
        points = []
        for idx in range(len(batch_images)):
            image_id = batch_ids[idx]
            img_byte_arr = batch_byte_arrs[idx]
            img_size = batch_sizes[idx]
            path = valid_paths[idx]
            vector = vectors[idx]
            
            minio_filename = f"{image_id}.webp"
            
            # Extract Cam and Frame if available in the filename
            cam = None
            frame = None
            payload = {}
            match = re.search(r"Vid_(\d+)_frame(\d+)", base_name, re.IGNORECASE)
            if match:
                cam = int(match.group(1))
                frame = int(match.group(2))
            
            
            if cam is not None:
                payload["Cam"] = cam
            if frame is not None:
                payload["Frame"] = frame
            
            try:
                minio_client.put_object(
                    BUCKET_NAME,
                    minio_filename,
                    img_byte_arr,
                    length=img_size,
                    content_type="image/webp"
                )
                points.append(PointStruct(
                    id=image_id,
                    vector=vector,
                    payload=payload
                ))
            except Exception as e:
                logger.error(f"Error uploading image {image_id} to MinIO: {e}")
                fail_count += 1
                continue
                
        # Upsert batch to Qdrant
        if points:
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                success_count += len(points)
            except Exception as e:
                logger.error(f"Error upserting batch to Qdrant: {e}")
                fail_count += len(points)
                
    logger.info(f"Ingestion complete. Successfully processed {success_count} images. Failed: {fail_count}.")

if __name__ == "__main__":
    ingest_dataset()
