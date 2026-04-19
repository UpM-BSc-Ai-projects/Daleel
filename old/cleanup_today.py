import os
from datetime import datetime, timezone
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Constants (matching your app.py environment/defaults)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.getenv("MINIO_BUCKET", "image-dataset")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "images")

def cleanup_today():
    # Initialize clients
    print("Connecting to MinIO and Qdrant...")
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Get local "today" date
    today = datetime.now(timezone.utc).date()
    print(f"Searching for items added on: {today}")

    # 1. List objects in MinIO to find those added today
    objects_to_delete = []
    try:
        objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            # Check if last_modified matches today
            if obj.last_modified.date() == today:
                # Get the image_id (filename without extension)
                image_id = obj.object_name.split('.')[0]
                objects_to_delete.append((obj.object_name, image_id))
    except Exception as e:
        print(f"Error listing MinIO objects: {e}")
        return

    if not objects_to_delete:
        print("No items found added today.")
        return

    print(f"Found {len(objects_to_delete)} items to delete.")
    
    ids_to_del = [item[1] for item in objects_to_delete]
    filenames_to_del = [item[0] for item in objects_to_delete]

    # 2. Delete from Qdrant
    print(f"Deleting {len(ids_to_del)} points from Qdrant collection '{COLLECTION_NAME}'...")
    try:
        # Use batch delete for IDs
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids_to_del
        )
        print("Successfully deleted from Qdrant.")
    except Exception as e:
        print(f"Error deleting from Qdrant: {e}")

    # 3. Delete from MinIO
    print("Deleting objects from MinIO...")
    try:
        for filename in filenames_to_del:
            minio_client.remove_object(BUCKET_NAME, filename)
        print("Successfully deleted from MinIO.")
    except Exception as e:
        print(f"Error deleting from MinIO: {e}")

    print("\nCleanup complete!")

if __name__ == "__main__":
    cleanup_today()
