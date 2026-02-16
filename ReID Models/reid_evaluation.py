"""
ReID Model Evaluation Script
Supports multiple dataset formats and ReID models from boxmot library
Evaluates using Rank-1, Rank-5, Rank-10, and mAP metrics
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import re
import json
from datetime import datetime

# BoxMOT ReID models
BOXMOT_AVAILABLE = False
ReidAutoBackend = None

try:
    # Use the official ReID class from boxmot.appearance.reid.auto_backend
    from boxmot.appearance.reid.auto_backend import ReidAutoBackend
    BOXMOT_AVAILABLE = True
    print("✓ Using boxmot.appearance.reid.auto_backend.ReidAutoBackend")
except ImportError as e:
    print(f"✗ Warning: boxmot ReID not available: {e}")
    print("  Install with: pip install boxmot")
    BOXMOT_AVAILABLE = False


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.name = self.dataset_path.name
        
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        Load gallery and query sets.
        
        Returns:
            Tuple of (gallery_dict, query_dict) where each dict maps:
            person_id -> list of (image_path, camera_id)
        """
        raise NotImplementedError
    
    @staticmethod
    def extract_id_from_filename(filename: str) -> Optional[int]:
        """Extract person ID from filename. Override in subclasses."""
        raise NotImplementedError


class AGReIDLoader(DatasetLoader):
    """Loader for AG-ReID.v2 dataset."""
    
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        AG-ReID structure:
        - val/gallery/ or val/gallery_flat/
        - val/query/ or val/query_flat/
        Filename format: 00000003_0002_00000007.jpg
        """
        gallery_dict = defaultdict(list)
        query_dict = defaultdict(list)
        
        # Try flat first (AG-ReID.v2 uses flat), then nested structures
        gallery_paths = [
            self.dataset_path / 'val' / 'gallery_flat',
            self.dataset_path / 'val' / 'gallery'
        ]
        query_paths = [
            self.dataset_path / 'val' / 'query_flat',
            self.dataset_path / 'val' / 'query'
        ]
        
        # Find existing gallery path
        gallery_dir = None
        for gp in gallery_paths:
            if gp.exists():
                gallery_dir = gp
                break
        
        # Find existing query path
        query_dir = None
        for qp in query_paths:
            if qp.exists():
                query_dir = qp
                break
        
        if not gallery_dir or not query_dir:
            raise ValueError(f"Could not find gallery/query directories in {self.dataset_path}")
        
        # Load gallery (support both jpg and png)
        for img_path in list(gallery_dir.glob('*.jpg')) + list(gallery_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None:
                gallery_dict[person_id].append((img_path, camera_id))
        
        # Load query (support both jpg and png)
        for img_path in list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None:
                query_dict[person_id].append((img_path, camera_id))
        
        return dict(gallery_dict), dict(query_dict)
    
    @staticmethod
    def extract_id_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract person ID and camera ID from AG-ReID filename.
        New format: P0001T03220A1C0F00091.jpg -> (person_id=1, camera_id=0)
        Old format: 00000003_0002_00000007.jpg -> (person_id=3, camera_id=2)
        Supports both .jpg and .png
        """
        # Try new AG-ReID format: P####T#####A#C#F#####
        match = re.match(r'P(\d{4})T\d+A\d+C(\d+)F\d+\.(jpg|png)', filename)
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            return person_id, camera_id
        
        # Try old format
        match = re.match(r'(\d+)_(\d+)_\d+\.(jpg|png)', filename)
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            return person_id, camera_id
        
        return None, None


class PDukeMTMCLoader(DatasetLoader):
    """Loader for P-DukeMTMC-reid dataset."""
    
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        P-DukeMTMC structure:
        - test/gallery (occluded) or test/gallery (whole)
        - test/query (whole)
        This dataset may have different ID encoding - needs inspection
        """
        gallery_dict = defaultdict(list)
        query_dict = defaultdict(list)
        
        # Check for test directory
        test_dir = self.dataset_path / 'test'
        if not test_dir.exists():
            raise ValueError(f"Test directory not found in {self.dataset_path}")
        
        # Find gallery directories
        gallery_dirs = [
            test_dir / 'gallery (occluded)',
            test_dir / 'gallery (whole)',
            test_dir / 'gallery'
        ]
        
        query_dirs = [
            test_dir / 'query (whole)',
            test_dir / 'query'
        ]
        
        # Load from all available gallery directories
        for gallery_dir in gallery_dirs:
            if gallery_dir.exists():
                for img_path in list(gallery_dir.glob('*.jpg')) + list(gallery_dir.glob('*.png')):
                    person_id, camera_id = self.extract_id_from_filename(img_path.name)
                    if person_id is not None:
                        gallery_dict[person_id].append((img_path, camera_id))
        
        # Load from query directory
        for query_dir in query_dirs:
            if query_dir.exists():
                for img_path in list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png')):
                    person_id, camera_id = self.extract_id_from_filename(img_path.name)
                    if person_id is not None:
                        query_dict[person_id].append((img_path, camera_id))
        
        if not gallery_dict or not query_dict:
            raise ValueError(f"Could not load gallery/query from {self.dataset_path}")
        
        return dict(gallery_dict), dict(query_dict)
    
    @staticmethod
    def extract_id_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract person ID from DukeMTMC-style filename.
        Common formats: 0001_c1_f0000001.jpg or similar
        Supports both .jpg and .png
        """
        # Try DukeMTMC format: XXXX_cY_fZZZZZZ.(jpg|png)
        match = re.match(r'(\d+)_c(\d+)_', filename)
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            return person_id, camera_id
        
        # Try simple format: XXXX_Y_Z.(jpg|png)
        match = re.match(r'(\d+)_(\d+)_', filename)
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            return person_id, camera_id
        
        return None, None


class PETHZLoader(DatasetLoader):
    """Loader for P_ETHZ dataset."""
    
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        P_ETHZ structure (no train/test subdirectories):
        - gallery (occluded)/ or gallery (whole)/ directly in root
        - query (whole)/ directly in root
        """
        gallery_dict = defaultdict(list)
        query_dict = defaultdict(list)
        
        # Find gallery directories (directly in dataset root, no test/ subdirectory)
        gallery_paths = [
            self.dataset_path / 'gallery (occluded)',
            self.dataset_path / 'gallery (whole)',
            self.dataset_path / 'gallery'
        ]
        
        query_paths = [
            self.dataset_path / 'query (whole)',
            self.dataset_path / 'query'
        ]
        
        # Find existing gallery paths
        gallery_dirs = [gp for gp in gallery_paths if gp.exists()]
        
        # Find existing query path
        query_dir = None
        for qp in query_paths:
            if qp.exists():
                query_dir = qp
                break
        
        if not gallery_dirs or not query_dir:
            raise ValueError(f"Could not find gallery/query directories in {self.dataset_path}")
        
        # Load gallery from all found gallery directories
        for gallery_dir in gallery_dirs:
            for img_path in list(gallery_dir.glob('*.jpg')) + list(gallery_dir.glob('*.png')):
                person_id, camera_id = self.extract_id_from_filename(img_path.name)
                if person_id is not None:
                    gallery_dict[person_id].append((img_path, camera_id))
        
        # Load query
        for img_path in list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None:
                query_dict[person_id].append((img_path, camera_id))
        
        return dict(gallery_dict), dict(query_dict)

    @staticmethod
    def extract_id_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract person ID from P_ETHZ filename.
        Format: 001_001.(jpg|png) -> (person_id=1, instance_id=1)
        Supports both .jpg and .png
        """
        match = re.match(r'(\d+)_(\d+)', filename)
        if match:
            person_id = int(match.group(1))
            instance_id = int(match.group(2))
            return person_id, None
        return None, None

class PRAI1581Loader(DatasetLoader):
    """Loader for PRAI-1581 dataset."""
    
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        PRAI-1581 structure:
        - val/gallery/
        - val/query/
        Filename format: 00000003_0002_00000007.jpg (same as AG-ReID)
        """
        gallery_dict = defaultdict(list)
        query_dict = defaultdict(list)
        
        gallery_dir = self.dataset_path / 'val' / 'gallery'
        query_dir = self.dataset_path / 'val' / 'query'
        
        if not gallery_dir.exists() or not query_dir.exists():
            raise ValueError(f"Gallery/query directories not found in {self.dataset_path}")
        
        # Load gallery
        for img_path in list(gallery_dir.glob('*.jpg')) + list(gallery_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None:
                gallery_dict[person_id].append((img_path, camera_id))
        
        # Load query
        for img_path in list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None:
                query_dict[person_id].append((img_path, camera_id))
        
        return dict(gallery_dict), dict(query_dict)
    
    @staticmethod
    def extract_id_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """Same format as AG-ReID."""
        return AGReIDLoader.extract_id_from_filename(filename)


class Market1501Loader(DatasetLoader):
    """Loader for Market-1501 dataset."""
    
    def load_gallery_query(self) -> Tuple[Dict, Dict]:
        """
        Market-1501 structure:
        - bounding_box_test/ (gallery)
        - query/
        Filename format: 0001_c1s1_000001_00.jpg
        """
        gallery_dict = defaultdict(list)
        query_dict = defaultdict(list)
        
        # Try different possible paths
        gallery_paths = [
            self.dataset_path / 'bounding_box_test',
            self.dataset_path / 'gallery',
            self.dataset_path / 'val' / 'gallery'
        ]
        
        query_paths = [
            self.dataset_path / 'query',
            self.dataset_path / 'val' / 'query'
        ]
        
        gallery_dir = None
        for gp in gallery_paths:
            if gp.exists():
                gallery_dir = gp
                break
        
        query_dir = None
        for qp in query_paths:
            if qp.exists():
                query_dir = qp
                break
        
        if not gallery_dir or not query_dir:
            raise ValueError(f"Gallery/query directories not found in {self.dataset_path}")
        
        # Load gallery
        for img_path in list(gallery_dir.glob('*.jpg')) + list(gallery_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None and person_id != -1:  # Market uses -1 for junk images
                gallery_dict[person_id].append((img_path, camera_id))
        
        # Load query
        for img_path in list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.png')):
            person_id, camera_id = self.extract_id_from_filename(img_path.name)
            if person_id is not None and person_id != -1:
                query_dict[person_id].append((img_path, camera_id))
        
        return dict(gallery_dict), dict(query_dict)
    
    @staticmethod
    def extract_id_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract person ID from Market-1501 filename.
        Format: 0001_c1s1_000001_00.(jpg|png) -> (person_id=1, camera_id=1)
        Supports both .jpg and .png
        """
        match = re.match(r'(\d+)_c(\d+)', filename)
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            return person_id, camera_id
        return None, None


class DatasetFactory:
    """Factory to create appropriate dataset loader."""
    
    LOADERS = {
        'ag-reid': AGReIDLoader,
        'agreid': AGReIDLoader,
        'ag_reid': AGReIDLoader,
        'p-dukemtmc': PDukeMTMCLoader,
        'pdukemtmc': PDukeMTMCLoader,
        'duke': PDukeMTMCLoader,
        'p_ethz': PETHZLoader,
        'pethz': PETHZLoader,
        'ethz': PETHZLoader,
        'prai-1581': PRAI1581Loader,
        'prai1581': PRAI1581Loader,
        'prai': PRAI1581Loader,
        'market1501': Market1501Loader,
        'market': Market1501Loader,
        'market-1501': Market1501Loader,
    }
    
    @classmethod
    def create_loader(cls, dataset_path: Path) -> DatasetLoader:
        """Create appropriate loader based on dataset path name."""
        dataset_name = dataset_path.name.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        # Try exact match first
        for key, loader_class in cls.LOADERS.items():
            key_normalized = key.lower().replace(' ', '').replace('-', '').replace('_', '')
            if key_normalized in dataset_name or dataset_name in key_normalized:
                print(f"Detected dataset type: {loader_class.__name__}")
                return loader_class(dataset_path)
        
        # Default to AG-ReID format if can't detect
        print(f"Warning: Could not auto-detect dataset type for {dataset_path.name}")
        print("Defaulting to AG-ReID format. Use --dataset-type to specify manually.")
        return AGReIDLoader(dataset_path)


class ReIDEvaluator:
    """Evaluate ReID models using standard metrics."""
    
    def __init__(self, model_path: str, device: str = 'cuda', half: bool = False):
        """
        Initialize ReID model for evaluation.
        
        Args:
            model_path: Path to ReID model weights
            device: Device to run inference on
            half: Whether to use half precision
        """
        if not BOXMOT_AVAILABLE:
            raise ImportError("boxmot library is required. Install with: pip install boxmot")
        
        if ReidAutoBackend is None:
            raise ImportError("boxmot.appearance.reid.auto_backend.ReidAutoBackend class could not be imported")
        
        self.device = '0' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        
        print(f"Loading ReID model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Initialize ReID model from boxmot
        # ReidAutoBackend signature: ReidAutoBackend(weights, device='cpu', half=False)
        # weights must be a Path object
        self.backend = ReidAutoBackend(
            weights=self.model_path,
            device=self.device,
            half=half
        )
        # Get the actual model backend for inference
        self.model = self.backend.get_backend()
        
        print(f"✓ Model loaded successfully!")
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector as numpy array
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # For ReID evaluation, we treat the whole image as the detection
        # Create a bounding box that covers the entire image
        h, w = img.shape[:2]
        # Format: (x1, y1, x2, y2) - only 4 values as expected by boxmot get_crops
        dets = np.array([[0, 0, w, h]], dtype=np.float32)
        
        # Extract features using boxmot ReID backend
        # get_features expects (xyxys, img) and returns normalized embeddings
        features = self.model.get_features(dets, img)
        
        if features.size > 0:
            features = features[0]  # Get first (and only) embedding
        else:
            raise ValueError(f"No features extracted from {image_path}")
        
        return features
    
    def compute_distance_matrix(self, query_features: np.ndarray, 
                               gallery_features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix between query and gallery.
        
        Args:
            query_features: Query feature matrix (N_query x feature_dim)
            gallery_features: Gallery feature matrix (N_gallery x feature_dim)
            
        Returns:
            Distance matrix (N_query x N_gallery)
        """
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(query_features, gallery_features.T)
        distance = 1 - similarity
        
        return distance
    
    def evaluate_dataset(self, dataset_loader: DatasetLoader, 
                        remove_same_camera: bool = True) -> Dict:
        """
        Evaluate ReID model on a dataset.
        
        Args:
            dataset_loader: Dataset loader instance
            remove_same_camera: Whether to remove gallery images from same camera as query
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on: {dataset_loader.name}")
        print(f"{'='*60}")
        
        # Load dataset
        gallery_dict, query_dict = dataset_loader.load_gallery_query()
        
        print(f"Gallery IDs: {len(gallery_dict)}")
        print(f"Query IDs: {len(query_dict)}")
        
        # Extract features for gallery
        print("\nExtracting gallery features...")
        gallery_features = []
        gallery_labels = []
        gallery_cameras = []
        gallery_paths = []
        
        for person_id, image_list in tqdm(gallery_dict.items()):
            for img_path, camera_id in image_list:
                try:
                    features = self.extract_features(img_path)
                    gallery_features.append(features)
                    gallery_labels.append(person_id)
                    gallery_cameras.append(camera_id)
                    gallery_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        gallery_features = np.array(gallery_features)
        gallery_labels = np.array(gallery_labels)
        gallery_cameras = np.array(gallery_cameras)
        
        print(f"Gallery size: {len(gallery_features)}")
        
        # Extract features for queries
        print("\nExtracting query features...")
        query_features = []
        query_labels = []
        query_cameras = []
        query_paths = []
        
        for person_id, image_list in tqdm(query_dict.items()):
            for img_path, camera_id in image_list:
                try:
                    features = self.extract_features(img_path)
                    query_features.append(features)
                    query_labels.append(person_id)
                    query_cameras.append(camera_id)
                    query_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        query_features = np.array(query_features)
        query_labels = np.array(query_labels)
        query_cameras = np.array(query_cameras)
        
        print(f"Query size: {len(query_features)}")
        
        # Compute distance matrix
        print("\nComputing distance matrix...")
        distance_matrix = self.compute_distance_matrix(query_features, gallery_features)
        
        # Compute metrics
        print("\nComputing evaluation metrics...")
        metrics = self.compute_metrics(
            distance_matrix,
            query_labels,
            gallery_labels,
            query_cameras if remove_same_camera else None,
            gallery_cameras if remove_same_camera else None
        )
        
        return metrics
    
    def compute_metrics(self, distance_matrix: np.ndarray,
                       query_labels: np.ndarray,
                       gallery_labels: np.ndarray,
                       query_cameras: Optional[np.ndarray] = None,
                       gallery_cameras: Optional[np.ndarray] = None) -> Dict:
        """
        Compute Rank-1, Rank-5, Rank-10, and mAP.
        
        Args:
            distance_matrix: Distance matrix (N_query x N_gallery)
            query_labels: Query person IDs
            gallery_labels: Gallery person IDs
            query_cameras: Query camera IDs (optional)
            gallery_cameras: Gallery camera IDs (optional)
            
        Returns:
            Dictionary with metrics
        """
        num_queries = len(query_labels)
        
        # For CMC curve
        cmc = np.zeros(len(gallery_labels))
        ap_scores = []
        
        for i in range(num_queries):
            # Get query info
            q_label = query_labels[i]
            q_camera = query_cameras[i] if query_cameras is not None else None
            
            # Get distances for this query
            distances = distance_matrix[i]
            
            # Create mask for valid gallery images
            if q_camera is not None and gallery_cameras is not None:
                # Remove gallery images from same camera
                valid_mask = gallery_cameras != q_camera
            else:
                valid_mask = np.ones(len(gallery_labels), dtype=bool)
            
            # Get valid gallery labels and distances
            valid_labels = gallery_labels[valid_mask]
            valid_distances = distances[valid_mask]
            
            # Sort by distance (ascending)
            sorted_indices = np.argsort(valid_distances)
            sorted_labels = valid_labels[sorted_indices]
            
            # Find matches
            matches = (sorted_labels == q_label)
            
            # CMC curve
            if np.any(matches):
                first_match_idx = np.where(matches)[0][0]
                cmc[first_match_idx:] += 1
            
            # Average Precision
            if np.any(matches):
                num_matches = np.sum(matches)
                match_indices = np.where(matches)[0]
                
                # Compute precision at each recall point
                precisions = []
                for rank, idx in enumerate(match_indices):
                    precision = (rank + 1) / (idx + 1)
                    precisions.append(precision)
                
                ap = np.mean(precisions)
                ap_scores.append(ap)
            else:
                ap_scores.append(0.0)
        
        # Compute final metrics
        cmc = cmc / num_queries
        mAP = np.mean(ap_scores)
        
        metrics = {
            'Rank-1': float(cmc[0] * 100),
            'Rank-5': float(cmc[4] * 100) if len(cmc) > 4 else 0.0,
            'Rank-10': float(cmc[9] * 100) if len(cmc) > 9 else 0.0,
            'mAP': float(mAP * 100)
        }
        
        return metrics


class MultiModelEvaluator:
    """Evaluate multiple ReID models on multiple datasets."""
    
    def __init__(self, model_paths: List[str], dataset_paths: List[str],
                 output_dir: str = 'reid_evaluation_results',
                 device: str = 'cuda', half: bool = False,
                 remove_same_camera: bool = True):
        """
        Initialize multi-model evaluator.
        
        Args:
            model_paths: List of paths to ReID model weights
            dataset_paths: List of paths to datasets
            output_dir: Directory to save results
            device: Device to run inference on
            half: Whether to use half precision
            remove_same_camera: Whether to remove same-camera matches
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.output_dir = Path(output_dir)
        self.device = device
        self.half = half
        self.remove_same_camera = remove_same_camera
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
    
    def evaluate_all(self):
        """Evaluate all model-dataset combinations."""
        print(f"\n{'='*80}")
        print(f"REID MODEL EVALUATION")
        print(f"{'='*80}")
        print(f"Models to evaluate: {len(self.model_paths)}")
        print(f"Datasets to evaluate: {len(self.dataset_paths)}")
        print(f"Total combinations: {len(self.model_paths) * len(self.dataset_paths)}")
        print(f"Remove same camera: {self.remove_same_camera}")
        print(f"{'='*80}\n")
        
        for model_path in self.model_paths:
            print(f"\n{'#'*80}")
            print(f"MODEL: {model_path.name}")
            print(f"{'#'*80}")
            
            # Initialize model
            try:
                evaluator = ReIDEvaluator(
                    model_path=str(model_path),
                    device=self.device,
                    half=self.half
                )
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue
            
            # Evaluate on each dataset
            for dataset_path in self.dataset_paths:
                try:
                    # Create dataset loader
                    loader = DatasetFactory.create_loader(dataset_path)
                    
                    # Evaluate
                    metrics = evaluator.evaluate_dataset(
                        loader,
                        remove_same_camera=self.remove_same_camera
                    )
                    
                    # Store results
                    result = {
                        'model': model_path.name,
                        'model_path': str(model_path),
                        'dataset': dataset_path.name,
                        'dataset_path': str(dataset_path),
                        'timestamp': datetime.now().isoformat(),
                        **metrics
                    }
                    self.results.append(result)
                    
                    # Print results
                    print(f"\nResults for {dataset_path.name}:")
                    print(f"  Rank-1:  {metrics['Rank-1']:.2f}%")
                    print(f"  Rank-5:  {metrics['Rank-5']:.2f}%")
                    print(f"  Rank-10: {metrics['Rank-10']:.2f}%")
                    print(f"  mAP:     {metrics['mAP']:.2f}%")
                    
                except Exception as e:
                    print(f"Error evaluating {dataset_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to CSV and JSON."""
        if not self.results:
            print("No results to save!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = self.output_dir / f"reid_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Save JSON
        json_path = self.output_dir / f"reid_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to: {json_path}")
        
        return csv_path, json_path
    
    def print_summary(self):
        """Print summary table of all results."""
        if not self.results:
            print("No results to summarize!")
            return
        
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        # Create summary table
        df = pd.DataFrame(self.results)
        
        # Group by model and compute average
        print("Average Performance by Model:")
        print("-" * 80)
        model_summary = df.groupby('model')[['Rank-1', 'Rank-5', 'Rank-10', 'mAP']].mean()
        print(model_summary.to_string())
        
        print("\n" + "-" * 80)
        print("Performance by Model-Dataset Combination:")
        print("-" * 80)
        
        # Pivot table for easy viewing
        for metric in ['Rank-1', 'Rank-5', 'Rank-10', 'mAP']:
            print(f"\n{metric}:")
            pivot = df.pivot(index='model', columns='dataset', values=metric)
            print(pivot.to_string())
        
        print(f"\n{'='*80}")


def main():
    """Main execution function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate ReID models on multiple datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, single dataset
  python reid_evaluation.py --model osnet_x1_0.pt --dataset /path/to/AG-ReID.v2
  
  # Multiple models, single dataset
  python reid_evaluation.py --model osnet_x1_0.pt resnet50.pt clip_reid.pt --dataset /path/to/AG-ReID.v2
  
  # Single model, multiple datasets
  python reid_evaluation.py --model osnet_x1_0.pt --dataset /path/to/AG-ReID.v2 /path/to/Market1501
  
  # Multiple models and datasets
  python reid_evaluation.py --model osnet_x1_0.pt resnet50.pt --dataset /path/to/AG-ReID.v2 /path/to/Market1501 /path/to/PRAI-1581
  
  # From model directory
  python reid_evaluation.py --model-dir /path/to/models/ --dataset /path/to/AG-ReID.v2
        """
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', nargs='+', help='Path(s) to ReID model weight file(s)')
    model_group.add_argument('--model-dir', help='Directory containing ReID model files')
    
    # Dataset arguments
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset', nargs='+', help='Path(s) to dataset(s)')
    dataset_group.add_argument('--dataset-dir', help='Directory containing multiple datasets')
    
    # Evaluation arguments
    parser.add_argument('--output-dir', default='reid_evaluation_results',
                       help='Directory to save results (default: reid_evaluation_results)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on (default: cuda)')
    parser.add_argument('--half', action='store_true',
                       help='Use half precision (FP16)')
    parser.add_argument('--keep-same-camera', action='store_true',
                       help='Keep gallery images from same camera as query (default: remove)')
    
    args = parser.parse_args()
    
    # Collect model paths
    model_paths = []
    if args.model:
        model_paths = args.model
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        # Find all model files
        for ext in ['.pt', '.pth', '.onnx', '.engine']:
            model_paths.extend([str(p) for p in model_dir.glob(f"*{ext}")])
        
        if not model_paths:
            raise ValueError(f"No model files found in {model_dir}")
    
    # Collect dataset paths
    dataset_paths = []
    if args.dataset:
        dataset_paths = args.dataset
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Get all subdirectories as datasets
        dataset_paths = [str(p) for p in dataset_dir.iterdir() if p.is_dir()]
        
        if not dataset_paths:
            raise ValueError(f"No dataset directories found in {dataset_dir}")
    
    # Print configuration
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print(f"{'='*80}")
    print(f"Models ({len(model_paths)}):")
    for mp in model_paths:
        print(f"  - {mp}")
    print(f"\nDatasets ({len(dataset_paths)}):")
    for dp in dataset_paths:
        print(f"  - {dp}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Half precision: {args.half}")
    print(f"Remove same camera: {not args.keep_same_camera}")
    print(f"{'='*80}\n")
    
    # Create evaluator
    evaluator = MultiModelEvaluator(
        model_paths=model_paths,
        dataset_paths=dataset_paths,
        output_dir=args.output_dir,
        device=args.device,
        half=args.half,
        remove_same_camera=not args.keep_same_camera
    )
    
    # Run evaluation
    evaluator.evaluate_all()


if __name__ == "__main__":
    # Check dependencies
    print("Checking dependencies...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    if not BOXMOT_AVAILABLE:
        print("\nERROR: boxmot library not found!")
        print("Install with: pip install boxmot")
        exit(1)
    
    if ReidAutoBackend is None:
        print("\nERROR: Could not import boxmot.appearance.reid.auto_backend.ReidAutoBackend!")
        print("Your boxmot installation might be incomplete.")
        print("Try: pip install --upgrade boxmot")
        exit(1)
    
    print(f"✓ Using boxmot.appearance.reid.auto_backend.ReidAutoBackend")
    
    print("\n")
    main()
