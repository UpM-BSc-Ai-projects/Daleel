# Demo/process_person_embeddings_worker.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch, os, numpy as np
from pathlib import Path
from transformers import pipeline
from Embedding_worker.helpers import extract_frame_by_count, get_person_crop_from_coords
from CLIP.clip_v2 import CLIP
from time import sleep
from database import SessionLocal
from models import CameraDetectedPerson
from crud import update_camera_detected_person

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "stream"
CLIP_COLLECTION = "CLIP_embeddings"

texts = [
    "an elderly person over 60 years old",
    "a person in a wheel chair"
]

def sync_clip_embeddings_to_sql(interval_seconds: int = 30):
    """
    Continuously sync CLIP embeddings from Qdrant to SQLAlchemy DB.
    Pulls all records from CLIP_embeddings collection and updates the SQL DB.
    """
    
    
    client = QdrantClient(QDRANT_URL)
    num_synced = 0
    while True:
        try:
            # Scroll all records from CLIP_embeddings
            records = client.retrieve(
                collection_name=CLIP_COLLECTION,
                with_payload=True,
                ids = range(num_synced,num_synced+1000)
            )
            num_synced += len(records)
            if not records:
                print("[sync] No records in CLIP_embeddings")
                sleep(interval_seconds)
                continue
            
            db = SessionLocal()
            
            for rec in records:
                payload = rec.payload or {}
                pid = payload.get("Pid")
                is_lost = payload.get("is_lost", False)
                is_elderly = payload.get("is_elderly", False)
                is_disabled = payload.get("is_disabled", False)
                
                # Check if already exists in SQL
                existing = db.query(CameraDetectedPerson).filter(
                    CameraDetectedPerson.cameraDetectedPersonId == pid
                ).first()
                
                if existing:
                    # Update existing
                    update_camera_detected_person(
                        db,
                        pid,
                        potentiallyLost=is_lost,
                        isElderly=is_elderly,
                        isDisabled=is_disabled
                    )
                    print(f"[sync] Updated CDP {pid}")
                else:
                    # Create new (use Qdrant ID as the CDP ID)
                    try:
                        db_rec = CameraDetectedPerson(
                            cameraDetectedPersonId=pid,
                            potentiallyLost=is_lost,
                            isElderly=is_elderly,
                            isDisabled=is_disabled
                        )
                        db.add(db_rec)
                        db.commit()
                        db.refresh(db_rec)
                        print(f"[sync] Created new CDP {pid}")
                    except Exception as e:
                        db.rollback()
                        print(f"[sync] Error creating CDP {pid}: {e}")
            
            db.close()
            print(f"[sync] Synced {len(records)} records")
            
        except Exception as e:
            print(f"[sync] Error: {e}")
        
        sleep(interval_seconds)

def determine_elderly(elderly_embedding, person_emb, threshold=0.19):
  
    cos_sim = np.dot(person_emb, elderly_embedding) # Because it's already normalized, cosine similarity is the dot product    
    if cos_sim > threshold:
        return True
    return False
        


def determine_disabled(disabled_embedding, person_emb, threshold=0.19):

    cos_sim = np.dot(person_emb, disabled_embedding) # Because it's already normalized, cosine similarity is the dot product
    
    if cos_sim > threshold:
        return True
    return False

def process_embeddings(client, 
                       clip, 
                       emotion_clf, 
                       vid_location_mapping,
                       elderly_embedding,
                       disabled_embeddings,
                       limit,
                       num_processed):
    
    records = client.retrieve(collection_name=COLLECTION_NAME,ids=range(num_processed,limit),
                            with_payload=True)
    num_processed += len(records)
    

    for rec in records:
        payload = rec.payload
        cam_id = payload["cam_id"]
        coords = payload["coords"]
        frame_count = payload["frame_count"]
        pid = payload["Pid"]

        video_path = vid_location_mapping.get(cam_id)
        if not video_path:
            continue
        frame, frame_w, frame_h = extract_frame_by_count(video_path, frame_count)
        if frame is None:
            continue
        crop_image = get_person_crop_from_coords(frame, coords, frame_w, frame_h)
        if crop_image is None:
            continue

        emotion_result = emotion_clf(crop_image, top_k=1)[0]
        is_lost = emotion_result["label"].lower() in ["sad", "fear", "angry"] and emotion_result["score"] > 0.6

        # Embeddings
        clip_emb = clip.encode_person_crop(crop_image).to('cpu').numpy()
        is_elderly = determine_elderly(elderly_embedding,clip_emb)
        is_disabled = determine_disabled(disabled_embeddings,clip_emb)


        # Upsert to target collections
        client.upsert(collection_name=CLIP_COLLECTION,
                      points=[PointStruct(id=rec.id,
                                        vector=clip_emb,
                                        payload={"Pid": pid,
                                                 "is_lost":is_lost,
                                                 "is_elderly":is_elderly,
                                                 "is_disabled":is_disabled})])
        

def process_embeddings_job(paths):
   

    client = QdrantClient(QDRANT_URL)

    # Load models once
    clip = CLIP(model_name="ViT-B/32", device=DEVICE)
    emotion_clf = pipeline("image-classification", model="trpakov/vit-face-expression",
                           device=0 if torch.cuda.is_available() else -1)
    
    elderly_embedding, disabled_embeddings = clip.encode_text(texts).to('cpu')
    num_processed = 0
    limit = num_processed+300
    vid_location_mapping = {idx:path for idx,path in enumerate(paths)}

    while True:
        sleep(5)
        process_embeddings(client, 
                       clip, 
                       emotion_clf, 
                       vid_location_mapping,
                       elderly_embedding,
                       disabled_embeddings,
                       limit,
                       num_processed)





