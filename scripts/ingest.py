import os
from dotenv import load_dotenv
import re
import uuid
import logging
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from pathlib import Path
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# BoxMOT imports
from boxmot.reid.core import ReID
from boxmot.reid.backbones.clip.make_model_clipreid import load_clip_to_cpu, TextEncoder
from boxmot.reid.backbones.clip.clip import clip

# RealESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
DATASET_DIR = os.getenv("DATASET_DIR", r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/Person_Crops/v10")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "image-dataset")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "images")
REID_MODEL_PATH = os.getenv("REID_MODEL", "models/clip_market1501.pt")
SR_MODEL_PATH = os.getenv("SR_MODEL", "models/RealESRGAN_x4plus.pth")

# Super-Resolution Configuration
USE_UPSCALER = True  # Set to False to disable upscaling

BATCH_SIZE = 500


# ============================================================
#  BOXMOT REID EXTRACTOR
# ============================================================

class BoxmotReIDExtractor:
    """Extract dual embeddings (1280-dim full + 512-dim semantic)."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        logger.info(f"{'=' * 70}")
        logger.info(f"LOADING REID MODEL")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Model: {Path(model_path).name}")
        logger.info(f"  Device: {device}")
        
        self.reid = ReID(
            weights=Path(model_path),
            device=torch.device(device),
            half=False,
        )
        self.backend = self.reid.model
        
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"  ✓ Output: 1280-dim full + 512-dim semantic")
        logger.info(f"{'=' * 70}")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from BGR image.
        
        Returns:
            proj_emb_512 or None
        """
        h, w = image.shape[:2]
        dets = np.array([[0, 0, w, h]], dtype=np.float32)
        
        crops = self.backend.get_crops(dets, image)
        crops = self.backend.inference_preprocess(crops)
        
        with torch.no_grad():
            raw = self.backend.forward(crops)
        
        raw = self.backend.inference_postprocess(raw)
        
        if raw is None or (isinstance(raw, np.ndarray) and raw.size == 0):
            return None
        
        raw = raw[0]  # shape: (1280,)
        feat_512 = raw[768:]
        
        proj_emb = feat_512 / (np.linalg.norm(feat_512) + 1e-8)
        full_emb = raw / (np.linalg.norm(raw) + 1e-8)
        
        return proj_emb
    
    def extract_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract embeddings from PIL Image.
        
        Returns:
            proj_emb_512 or None
        """
        # Convert PIL to BGR numpy array
        img_rgb = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return self.extract(img_bgr)


# ============================================================
#  REALESRGAN UPSCALER
# ============================================================

class RealESRGANUpscaler:
    """Super-resolution upscaler using RealESRGAN."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        
        logger.info(f"{'=' * 70}")
        logger.info(f"LOADING SUPER-RESOLUTION MODEL")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Model: RealESRGAN_x4plus")
        logger.info(f"  Weights: {Path(model_path).name}")
        logger.info(f"  Device: {device}")
        
        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )
        
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"{'=' * 70}")
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale a BGR image.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Upscaled BGR image
        """
        output, _ = self.upsampler.enhance(image, outscale=4)
        return output
    
    def upscale_pil(self, pil_image: Image.Image) -> Image.Image:
        """
        Upscale a PIL Image.
        
        Args:
            pil_image: PIL Image
        
        Returns:
            Upscaled PIL Image
        """
        # Convert PIL to BGR
        img_rgb = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Upscale
        upscaled_bgr = self.upscale(img_bgr)
        
        # Convert back to PIL
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(upscaled_rgb)


# ============================================================
#  INITIALIZATION
# ============================================================

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
    
    # Initialize BoxMOT ReID Extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Upscaler enabled: {USE_UPSCALER}")
    
    reid_extractor = BoxmotReIDExtractor(REID_MODEL_PATH, device)
    
    # Initialize upscaler if enabled
    upscaler = None
    if USE_UPSCALER:
        if os.path.exists(SR_MODEL_PATH):
            upscaler = RealESRGANUpscaler(SR_MODEL_PATH, device)
        else:
            logger.warning(f"SR model not found at {SR_MODEL_PATH}. Upscaling disabled.")
    
    # Vector size is always 512 for semantic embeddings
    vector_size = 512
    
    collections = qdrant_client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME} with vector size {vector_size}")
    else:
        logger.info(f"Qdrant collection {COLLECTION_NAME} already exists.")
        
    return minio_client, qdrant_client, reid_extractor, upscaler


# ============================================================
#  INGESTION
# ============================================================

def ingest_dataset():
    """Main ingestion loop to process all images in the dataset directory using batches."""
    minio_client, qdrant_client, reid_extractor, upscaler = initialize_clients()
    
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
        
        batch_data = []
        
        # Load and compress images
        for path in batch_paths:
            try:
                base_name = os.path.splitext(os.path.basename(path))[0]
                image_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, base_name))
                
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Make copies for different purposes
                    img_original = img.copy()
                    
                    # Scale down to 90% for storage (to save space)
                    w, h = img.size
                    new_w = int(w * 0.9)
                    new_h = int(h * 0.9)
                    if new_w > 0 and new_h > 0:
                        img_for_storage = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        img_for_storage = img_original.copy()
                    
                    # Decide which image to use for embedding extraction
                    if USE_UPSCALER and upscaler is not None:
                        # Upscale for better embeddings
                        img_for_embedding = upscaler.upscale_pil(img_original)
                    else:
                        # Use original
                        img_for_embedding = img_original
                    
                    # Prepare storage image (WEBP compressed)
                    img_byte_arr = BytesIO()
                    img_for_storage.save(img_byte_arr, format='WEBP', quality=60)
                    img_byte_arr.seek(0)
                    img_size = img_byte_arr.getbuffer().nbytes
                    
                    batch_data.append({
                        'path': path,
                        'image_id': image_id,
                        'base_name': base_name,
                        'img_for_embedding': img_for_embedding,
                        'img_byte_arr': img_byte_arr,
                        'img_size': img_size,
                    })
                    
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                fail_count += 1
                
        if not batch_data:
            continue
            
        # Extract embeddings for batch
        logger.info(f"  Extracting embeddings for {len(batch_data)} images...")
        vectors = []
        valid_data = []
        
        for data in batch_data:
            try:
                # Extract 512-dim semantic embedding using BoxMOT ReID
                vector_512 = reid_extractor.extract_from_pil(data['img_for_embedding'])
                
                if vector_512 is not None:
                    vectors.append(vector_512.tolist())
                    valid_data.append(data)
                else:
                    logger.error(f"Failed to extract embedding for {data['path']}")
                    fail_count += 1
            except Exception as e:
                logger.error(f"Error extracting embedding for {data['path']}: {e}")
                fail_count += 1
                
        if not valid_data:
            continue
            
        # Upload to MinIO and prepare Qdrant points
        logger.info(f"  Uploading {len(valid_data)} images to MinIO and Qdrant...")
        points = []
        
        for idx, data in enumerate(valid_data):
            image_id = data['image_id']
            img_byte_arr = data['img_byte_arr']
            img_size = data['img_size']
            base_name = data['base_name']
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
                logger.info(f"  ✓ Successfully processed {len(points)} images in this batch")
            except Exception as e:
                logger.error(f"Error upserting batch to Qdrant: {e}")
                fail_count += len(points)
                
    logger.info(f"{'=' * 70}")
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Successfully processed: {success_count} images")
    logger.info(f"  Failed: {fail_count} images")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    ingest_dataset()