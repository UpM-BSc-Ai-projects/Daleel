import os
import logging
from dotenv import load_dotenv
from minio import Minio
from minio.deleteobjects import DeleteObject
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "image-dataset")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "images")

def clear_minio():
    logger.info(f"Connecting to MinIO at {MINIO_ENDPOINT}...")
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        if not minio_client.bucket_exists(BUCKET_NAME):
            logger.info(f"Bucket '{BUCKET_NAME}' does not exist in MinIO.")
            return

        logger.info(f"Clearing objects in MinIO bucket '{BUCKET_NAME}'...")
        
        # Get all objects
        objects_to_delete = minio_client.list_objects(BUCKET_NAME, recursive=True)
        delete_object_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        
        if not delete_object_list:
            logger.info(f"MinIO bucket '{BUCKET_NAME}' is already empty.")
            return

        # Delete objects
        errors = minio_client.remove_objects(BUCKET_NAME, delete_object_list)
        error_count = 0
        for error in errors:
            logger.error(f"Error deleting object: {error}")
            error_count += 1
            
        if error_count == 0:
            logger.info(f"Successfully deleted {len(delete_object_list)} objects from MinIO bucket '{BUCKET_NAME}'.")
        else:
            logger.warning(f"Finished with {error_count} errors while deleting from MinIO.")
            
    except Exception as e:
        logger.error(f"Failed to clear MinIO: {e}")

def clear_qdrant():
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        collections = qdrant_client.get_collections().collections
        if any(c.name == COLLECTION_NAME for c in collections):
            logger.info(f"Deleting Qdrant collection '{COLLECTION_NAME}'...")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Successfully deleted Qdrant collection '{COLLECTION_NAME}'.")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' does not exist in Qdrant.")
            
    except Exception as e:
        logger.error(f"Failed to clear Qdrant: {e}")

def main():
    logger.info("Starting database clearance...")
    clear_minio()
    clear_qdrant()
    logger.info("Database clearance complete.")

if __name__ == "__main__":
    main()
