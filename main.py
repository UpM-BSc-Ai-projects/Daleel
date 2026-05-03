import os
from dotenv import load_dotenv
import uuid
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, PointStruct, Range
import cv2
import time
from ultralytics import YOLO
from openai import OpenAI
from fastapi import FastAPI, HTTPException, WebSocket, File, Form, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import threading
import json
import asyncio
from typing import List
from pathlib import Path
import httpx

# BoxMOT imports
from boxmot.reid.core import ReID
from boxmot.reid.backbones.clip.make_model_clipreid import load_clip_to_cpu, TextEncoder
from boxmot.reid.backbones.clip.clip import clip

# RealESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

load_dotenv()

# ============================================================
#  CONFIGURATION
# ============================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "image-dataset")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "images")
REID_MODEL_PATH = os.getenv("REID_MODEL", "models/clip_market1501.pt")
SR_MODEL_PATH = os.getenv("SR_MODEL", "models/RealESRGAN_x4plus.pth")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL", "models/yolo_person_yv11_best.pt")
CAMERAS_FOLDER = os.getenv("CAMERAS_FOLDER", "cameras")

# Super-Resolution Configuration
USE_UPSCALER = True  # Set to False to disable upscaling

# Global instances
client_minio = None
client_qdrant = None
text_encoder = None
reid_extractor = None
upscaler = None
model_yolo = None

# Global state for capturing video
capture_active = False
capture_logs = []
capture_websockets = []


# ============================================================
#  BOXMOT TEXT ENCODER
# ============================================================

class BoxmotTextEncoder:
    """Text encoder using fine-tuned CLIP-ReID weights."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        print(f"\n{'=' * 70}")
        print(f"LOADING TEXT ENCODER (CLIP ViT-B/16)")
        print(f"{'=' * 70}")
        print(f"  Model: {Path(model_path).name}")
        print(f"  Device: {device}")
        
        # Load base CLIP model structure
        clip_model = load_clip_to_cpu(
            backbone_name="ViT-B-16",
            h_resolution=16,
            w_resolution=8,
            vision_stride_size=16,
        )
        if device == "cpu":
            clip_model.float()
        clip_model.to(device)
        self.clip_model = clip_model
        
        # Build TextEncoder
        text_encoder = TextEncoder(clip_model)
        
        # Load fine-tuned weights
        state_dict = torch.load(model_path, map_location="cpu")
        text_enc_weights = {
            k[len("text_encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.")
        }
        
        if text_enc_weights:
            text_encoder.load_state_dict(text_enc_weights, strict=True)
            print(f"  ✓ Loaded fine-tuned text encoder weights")
        else:
            print(f"  ℹ Using base CLIP weights (no fine-tuning found)")
        
        text_encoder.eval().to(device)
        self.text_encoder = text_encoder
        
        print(f"  ✓ Produces 512-dim embeddings")
        print(f"{'=' * 70}\n")
    
    def extract_text(self, text: str) -> np.ndarray:
        """Extract 512-dim text embedding."""
        with torch.no_grad():
            tokenized = clip.tokenize([text]).to(self.device)
            token_emb = self.clip_model.token_embedding(tokenized).type(
                self.clip_model.dtype
            )
            features = self.text_encoder(token_emb, tokenized)
            features = F.normalize(features.float(), dim=-1)
            return features.cpu().numpy()[0]


# ============================================================
#  BOXMOT REID EXTRACTOR
# ============================================================

class BoxmotReIDExtractor:
    """Extract dual embeddings (1280-dim full + 512-dim semantic)."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        print(f"\n{'=' * 70}")
        print(f"LOADING REID MODEL")
        print(f"{'=' * 70}")
        print(f"  Model: {Path(model_path).name}")
        print(f"  Device: {device}")
        
        self.reid = ReID(
            weights=Path(model_path),
            device=torch.device(device),
            half=False,
        )
        self.backend = self.reid.model
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Output: 1280-dim full + 512-dim semantic")
        print(f"{'=' * 70}\n")
    
    def extract(self, image: np.ndarray) -> tuple:
        """
        Extract embeddings from BGR image.
        
        Returns:
            (full_emb_1280, proj_emb_512) or (None, None)
        """
        h, w = image.shape[:2]
        dets = np.array([[0, 0, w, h]], dtype=np.float32)
        
        crops = self.backend.get_crops(dets, image)
        crops = self.backend.inference_preprocess(crops)
        
        with torch.no_grad():
            raw = self.backend.forward(crops)
        
        raw = self.backend.inference_postprocess(raw)
        
        if raw is None or (isinstance(raw, np.ndarray) and raw.size == 0):
            return None, None
        
        raw = raw[0]  # shape: (1280,)
        feat_512 = raw[768:]
        
        proj_emb = feat_512 / (np.linalg.norm(feat_512) + 1e-8)
        full_emb = raw / (np.linalg.norm(raw) + 1e-8)
        
        return proj_emb
    
    def extract_from_pil(self, pil_image: Image.Image) -> tuple:
        """
        Extract embeddings from PIL Image.
        
        Returns:
            (full_emb_1280, proj_emb_512) or (None, None)
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
        
        print(f"\n{'=' * 70}")
        print(f"LOADING SUPER-RESOLUTION MODEL")
        print(f"{'=' * 70}")
        print(f"  Model: RealESRGAN_x4plus")
        print(f"  Weights: {Path(model_path).name}")
        print(f"  Device: {device}")
        
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
        
        print(f"  ✓ Model loaded successfully")
        print(f"{'=' * 70}\n")
    
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
#  FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_minio, client_qdrant, text_encoder, reid_extractor, upscaler, model_yolo
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'=' * 70}")
    print(f"INITIALIZING MODELS")
    print(f"{'=' * 70}")
    print(f"  Device: {device}")
    print(f"  ReID Model: {REID_MODEL_PATH}")
    print(f"  Upscaler Enabled: {USE_UPSCALER}")
    print(f"{'=' * 70}\n")
    
    # Load BoxMOT models
    text_encoder = BoxmotTextEncoder(REID_MODEL_PATH, device)
    reid_extractor = BoxmotReIDExtractor(REID_MODEL_PATH, device)
    
    # Load upscaler if enabled
    if USE_UPSCALER:
        if os.path.exists(SR_MODEL_PATH):
            upscaler = RealESRGANUpscaler(SR_MODEL_PATH, device)
        else:
            print(f"WARNING: SR model not found at {SR_MODEL_PATH}. Upscaling disabled.")
            upscaler = None
    
    print("Loading MinIO client...")
    client_minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    print("Loading Qdrant client...")
    client_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        model_yolo = YOLO(YOLO_MODEL_PATH)
    else:
        print("YOLO file not found!")
    
    print(f"\n{'=' * 70}")
    print("ALL MODELS LOADED SUCCESSFULLY")
    print(f"{'=' * 70}\n")
    
    yield


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  WEBSOCKET & VIDEO CAPTURE
# ============================================================

async def broadcast_ws_message(msg: dict):
    for ws in list(capture_websockets):
        try:
            await ws.send_json(msg)
        except Exception:
            capture_websockets.remove(ws)


def process_video_loop(video_path: str, interval: float, yolo_conf: float):
    global capture_active
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        asyncio.run(broadcast_ws_message({"type": "error", "msg": f"Could not open video file."}))
        capture_active = False
        return
        
    asyncio.run(broadcast_ws_message({"type": "info", "msg": "System active. Monitoring for persons..."}))
    
    last_processed_msec = - (interval * 1000)
    
    while capture_active:
        target_msec = last_processed_msec + (interval * 1000)
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        
        ret, frame = cap.read()
        if not ret:
            asyncio.run(broadcast_ws_message({"type": "warn", "msg": "Reached end of video file."}))
            break
            
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = int(current_msec / 1000)
        last_processed_msec = current_msec
        
        asyncio.run(broadcast_ws_message({"type": "info", "msg": f"**Processing Video Time:** {current_sec}s"}))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        
        if model_yolo:
            results = model_yolo(frame, classes=[0], verbose=False, conf=yolo_conf)
            found_people = False
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    found_people = True
                    asyncio.run(broadcast_ws_message({"type": "success", "msg": f"[{current_sec}s] Detected {len(boxes)} person(s)!"}))
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        crop_original = pil_frame.crop((x1, y1, x2, y2))
                        
                        # Decide which image to use for embedding extraction
                        if USE_UPSCALER and upscaler is not None:
                            # Upscale the crop for better embeddings
                            crop_for_embedding = upscaler.upscale_pil(crop_original)
                        else:
                            # Use original crop
                            crop_for_embedding = crop_original
                        
                        # Extract 512-dim semantic embedding using BoxMOT ReID
                        vector_512 = reid_extractor.extract_from_pil(crop_for_embedding)
                        
                        if vector_512 is None:
                            continue
                        
                        image_id = str(uuid.uuid4())
                        
                        # Save ORIGINAL (non-upscaled) image to MinIO to save space
                        img_byte_arr = BytesIO()
                        crop_original.save(img_byte_arr, format='WEBP', quality=60)
                        img_byte_arr.seek(0)
                        img_size = img_byte_arr.getbuffer().nbytes
                        
                        client_minio.put_object(
                            BUCKET_NAME,
                            f"{image_id}.webp",
                            img_byte_arr,
                            length=img_size,
                            content_type="image/webp"
                        )
                        
                        # Store 512-dim embedding in Qdrant
                        client_qdrant.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[PointStruct(
                                id=image_id,
                                vector=vector_512.tolist(),
                                payload={"Cam": "Live", "Frame": f"Time_{current_sec}s", "timestamp": current_sec, "session": "active"}
                            )]
                        )
                        
                        asyncio.run(broadcast_ws_message({"type": "crop", "id": image_id, "sec": current_sec, "person": i+1}))
            
            if not found_people:
                asyncio.run(broadcast_ws_message({"type": "info", "msg": f"[{current_sec}s] Scanning... No one found."}))
                
        time.sleep(0.1)
        
    cap.release()
    capture_active = False


@app.websocket("/api/ws/capture")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    capture_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        if websocket in capture_websockets:
            capture_websockets.remove(websocket)


@app.post("/api/capture/start")
def start_capture(file: UploadFile = File(...), interval: float = Form(5.0), conf: float = Form(0.5)):
    global capture_active
    if capture_active:
        return JSONResponse({"status": "error", "msg": "Capture already running"})
        
    tmp_path = f"tmp_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(file.file.read())
        
    capture_active = True
    threading.Thread(target=process_video_loop, args=(tmp_path, interval, conf)).start()
    return {"status": "ok"}


@app.post("/api/capture/stop")
def stop_capture():
    global capture_active
    capture_active = False
    return {"status": "ok"}


# ============================================================
#  API ENDPOINTS
# ============================================================

@app.get("/api/stats")
def get_stats():
    stats = {
        'qdrant_points': 0,
        'minio_objects': 0,
        'qdrant_status': 'Offline',
        'minio_status': 'Offline',
        'minio_size_bytes': 0,
        'qdrant_est_bytes': 0,
        'upscaler_enabled': USE_UPSCALER,
    }
    
    try:
        col = client_qdrant.get_collection(COLLECTION_NAME)
        stats['qdrant_points'] = col.points_count
        stats['qdrant_status'] = 'Online'
        stats['qdrant_est_bytes'] = col.points_count * 512 * 4
    except Exception:
        pass

    try:
        objs = list(client_minio.list_objects(BUCKET_NAME))
        stats['minio_objects'] = len(objs)
        stats['minio_size_bytes'] = sum(obj.size for obj in objs)
        stats['minio_status'] = 'Online'
    except Exception:
        pass
            
    return stats


@app.get("/api/cameras/list")
def list_cameras():
    """List all camera folders"""
    if not os.path.exists(CAMERAS_FOLDER):
        return []
    try:
        cameras = [
            d for d in os.listdir(CAMERAS_FOLDER)
            if os.path.isdir(os.path.join(CAMERAS_FOLDER, d))
        ]
        return sorted(cameras)
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return []


@app.get("/api/cameras/{camera_name}/images")
def list_camera_images(camera_name: str):
    """List all images in a camera folder"""
    camera_path = os.path.join(CAMERAS_FOLDER, camera_name)
    if not os.path.exists(camera_path) or not os.path.isdir(camera_path):
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        images = [
            f for f in os.listdir(camera_path)
            if os.path.isfile(os.path.join(camera_path, f)) and 
            f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))
        ]
        return sorted(images, reverse=True)
    except Exception as e:
        print(f"Error listing images for camera {camera_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras/{camera_name}/image/{image_filename}")
def get_camera_image(camera_name: str, image_filename: str):
    """Serve an image from a camera folder"""
    image_path = os.path.join(CAMERAS_FOLDER, camera_name, image_filename)
    
    real_path = os.path.realpath(image_path)
    real_camera_path = os.path.realpath(os.path.join(CAMERAS_FOLDER, camera_name))
    if not real_path.startswith(real_camera_path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        
        ext = os.path.splitext(image_filename)[1].lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }
        content_type = content_type_map.get(ext, 'image/jpeg')
        
        return Response(content=img_data, media_type=content_type)
    except Exception as e:
        print(f"Error serving image {image_filename} from camera {camera_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def process_camera_image(camera_name: str, image_filename: str):
    """Background worker to index camera image - same pattern as process_video_loop"""
    image_path = os.path.join(CAMERAS_FOLDER, camera_name, image_filename)
    
    real_path = os.path.realpath(image_path)
    real_camera_path = os.path.realpath(os.path.join(CAMERAS_FOLDER, camera_name))
    if not real_path.startswith(real_camera_path):
        asyncio.run(broadcast_ws_message({"type": "error", "msg": f"Access denied: {image_filename}"}))
        return
    
    if not os.path.exists(image_path):
        asyncio.run(broadcast_ws_message({"type": "error", "msg": f"Image not found: {image_filename}"}))
        return
    
    try:
        asyncio.run(broadcast_ws_message({"type": "info", "msg": f"Indexing {image_filename} from {camera_name}..."}))
        
        img = Image.open(image_path).convert('RGB')
        vector = model_clip.encode(img).tolist()
        
        image_id = str(uuid.uuid4())
        
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='WEBP', quality=85)
        img_byte_arr.seek(0)
        img_size = img_byte_arr.getbuffer().nbytes
        
        client_minio.put_object(
            BUCKET_NAME,
            f"{image_id}.webp",
            img_byte_arr,
            length=img_size,
            content_type="image/webp"
        )
        
        client_qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=image_id,
                vector=vector,
                payload={"Cam": camera_name, "Frame": image_filename, "session": "camera"}
            )]
        )
        
        asyncio.run(broadcast_ws_message({"type": "success", "msg": f"Indexed {image_filename} from {camera_name}"}))
    except Exception as e:
        print(f"Error indexing image {image_filename} from camera {camera_name}: {e}")
        asyncio.run(broadcast_ws_message({"type": "error", "msg": f"Failed to index {image_filename}: {str(e)}"}))


@app.post("/api/cameras/{camera_name}/image/{image_filename}/index")
def index_camera_image(camera_name: str, image_filename: str):
    """Start background indexing of camera image (processes like video frames)"""
    try:
        # Start background processing thread, similar to video capture
        threading.Thread(target=process_camera_image, args=(camera_name, image_filename)).start()
        return {"status": "ok", "message": f"Indexing {image_filename} in background..."}
    except Exception as e:
        print(f"Error starting index job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras")
def get_cameras():
    cams = set()
    try:
        offset = None
        for _ in range(5):
            res, offset = client_qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["Cam"],
                with_vectors=False,
                offset=offset
            )
            for p in res:
                if "Cam" in p.payload and p.payload["Cam"] is not None:
                    cams.add(p.payload["Cam"])
            if offset is None:
                break
    except Exception:
        pass
    return sorted(list(cams), key=str)


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...), lang: str = Form("en")):
    if not MISTRAL_API_KEY:
        raise HTTPException(status_code=500, detail="Mistral API key not configured")
    
    audio_data = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            files = {'file': (file.filename, audio_data, file.content_type or 'audio/wav')}
            data = {
                'model': 'voxtral-mini-latest',
                'language': lang
            }
            headers = {'Authorization': f'Bearer {MISTRAL_API_KEY}'}
            
            response = await client.post(
                "https://api.mistral.ai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=60.0
            )
            
            if response.status_code != 200:
                print(f"Mistral error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Error from Mistral STT API")
            
            result = response.json()
            return {"text": result.get("text", "")}
            
        except Exception as e:
            print(f"STT Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
def search_endpoint(
    text_query: str = Form(""),
    recursive_id: str = Form(""),
    cameras: str = Form(""),
    frames: str = Form(""),
    score_threshold: float = Form(0.0),
    limit: int = Form(20),
    from_time: float = Form(None),
    to_time: float = Form(None),
    files: List[UploadFile] = File([])
):
    """
    Multi-modal search endpoint using BoxMOT ReID models.
    
    Supports:
    - Text queries (512-dim from BoxmotTextEncoder)
    - Image queries (512-dim semantic from BoxmotReIDExtractor)
    - Recursive queries (from previous results)
    - Multi-query fusion (mean of embeddings)
    
    Uses upscaled embeddings if USE_UPSCALER is True.
    """
    vectors = []
    
    # Process text query
    if text_query:
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in text_query)
        
        if is_arabic:
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    print("Warning: GEMINI_API_KEY not found in .env. Falling back to original query.")
                    text_vector = text_encoder.extract_text(text_query)
                else:
                    client_openai = OpenAI(
                        api_key=gemini_api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                    )
                    
                    response = client_openai.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {"role": "system", "content": "You are a professional translation assistant. Translate the following Arabic text to English. Output only the English translation without any extra formatting or conversational text."},
                            {"role": "user", "content": text_query}
                        ]
                    )
                    translated_query = response.choices[0].message.content.strip()
                    text_vector = text_encoder.extract_text(translated_query)
            except Exception as e:
                print(f"Translation error: {e}")
                text_vector = text_encoder.extract_text(text_query)
        else:
            # Use BoxmotTextEncoder for text
            text_vector = text_encoder.extract_text(text_query)
            
        vectors.append(text_vector)
    
    # Process recursive query
    if recursive_id:
        res = client_qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[recursive_id],
            with_vectors=True
        )
        if res and res[0].vector:
            vectors.append(res[0].vector)
    
    # Process image queries
    elif files:
        for file in files:
            if file.filename:
                img_bytes = file.file.read()
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
            
                # Use BoxmotReIDExtractor for images (512-dim semantic)
                img_vector = reid_extractor.extract_from_pil(img)
                
                if img_vector is not None:
                    vectors.append(img_vector)
    
    if not vectors:
        raise HTTPException(status_code=400, detail="No valid query provided")
    
    # Apply Mean fusion for multiple queries (OPTIMAL TECHNIQUE)
    vectors = [np.array(v) for v in vectors]  # ensure all vectors are numpy arrays first
    
    if len(vectors) > 1:        
        # Mean fusion
        query_vector = np.mean(vectors, axis=0)
        
        # Re-normalize the fused vector
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        query_vector = query_vector.tolist()
    else:
        query_vector = vectors[0] if isinstance(vectors[0], list) else vectors[0].tolist()
    
    # Build filter conditions
    filter_conditions = []
    
    def parse_value(v):
        v = v.strip()
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    if cameras:
        cam_list = [parse_value(c) for c in cameras.split(",") if c.strip()]
        if cam_list:
            filter_conditions.append(FieldCondition(key="Cam", match=MatchAny(any=cam_list)))
            
    if frames:
        frame_list = [parse_value(c) for c in frames.split(",") if c.strip()]
        if frame_list:
            filter_conditions.append(FieldCondition(key="Frame", match=MatchAny(any=frame_list)))


    if from_time is not None or to_time is not None:
        filter_conditions.append(FieldCondition(key="Frame", range=Range(gte=from_time, lte=to_time)))

    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    thresh = score_threshold if score_threshold > 0.0 else None

    # Execute search
    try:
        search_response = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=thresh
        )
        
        results = []
        for r in search_response.points:
            results.append({
                "id": r.id,
                "score": r.score,
                "cam": r.payload.get("Cam", "N/A"),
                "frame": r.payload.get("Frame", "N/A")
            })
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    obj_name = image_id
    if not obj_name.endswith(".webp"):
        obj_name += ".webp"
        
    try:
        response = client_minio.get_object(BUCKET_NAME, obj_name)
        img_data = response.read()
        response.close()
        response.release_conn()
        return Response(content=img_data, media_type="image/webp")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/api/session")
def delete_session(image_ids: str = Form(...)):  # comma separated
    ids = [i.strip() for i in image_ids.split(",") if i.strip()]
    if not ids:
        return {"status": "ok", "deleted": 0}
        
    try:
        client_qdrant.delete(collection_name=COLLECTION_NAME, points_selector=ids)
        for image_id in ids:
            try:
                client_minio.remove_object(BUCKET_NAME, f"{image_id}.webp")
            except Exception:
                pass
        return {"status": "ok", "deleted": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#  STATIC FILES
# ============================================================

# Ensure fallback for when frontend is not built
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
