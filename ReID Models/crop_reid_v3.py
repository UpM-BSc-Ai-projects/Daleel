"""
Person Detection, ReID Tracking, and Cropping Script
- Multiple configurable detection models
- NMS across all model detections
- Area and aspect ratio filtering
- Configurable BoxMOT tracker (with or without ReID)
- Configurable half-precision flag
- Auto-versioned output directories
"""

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
import boxmot


# ------------------------------------------------------------
#  Tracker factory
# ------------------------------------------------------------

# Maps lowercase tracker name -> boxmot class
TRACKER_MAP = {
    'botsort'    : boxmot.BotSort,
    'boosttrack' : boxmot.BoostTrack,
    'strongsort' : boxmot.StrongSort,
    'deepocsort' : boxmot.DeepOcSort,
    'hybridsort' : boxmot.HybridSort,
    'bytetrack'  : boxmot.ByteTrack,
    'ocsort'     : boxmot.OcSort,
}

# Trackers that accept reid_weights and half arguments
TRACKERS_WITH_REID = {'botsort', 'boosttrack', 'strongsort', 'deepocsort', 'hybridsort'}
# Motion-only trackers (no reid_weights / half)
TRACKERS_MOTION_ONLY = {'bytetrack', 'ocsort'}


def build_tracker(
    tracker_name: str,
    reid_weights: Path,
    device: torch.device,
    half: bool,
    det_thresh: float = 0.6,
    max_age: int = 60,
    min_hits: int = 3,
):
    """
    Instantiate the requested BoxMOT tracker.

    Args:
        tracker_name : One of the keys in TRACKER_MAP.
        reid_weights : Path to ReID model weights (ignored for motion-only trackers).
        device       : torch.device to run inference on.
        half         : Use FP16 for ReID model (ignored for motion-only trackers).
        det_thresh   : Minimum confidence threshold for tracker (default 0.6).
        max_age      : Maximum frames to keep a track alive without detections (default 60).
        min_hits     : Minimum consecutive detections before track confirmation (default 3).

    Returns:
        Instantiated tracker object.
    """
    name = tracker_name.lower()
    if name not in TRACKER_MAP:
        valid = ', '.join(sorted(TRACKER_MAP.keys()))
        raise ValueError(f"Unknown tracker '{tracker_name}'. Valid options: {valid}")

    cls = TRACKER_MAP[name]

    if name in TRACKERS_WITH_REID:
        return cls(
            reid_weights=reid_weights,
            device=device,
            half=half,
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
        )
    else:
        # Motion-only trackers only need device + tracking params
        return cls(
            device=device,
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
        )


# ------------------------------------------------------------
#  Detection model loader
# ------------------------------------------------------------

def load_model(model_path: str, model_type: str):
    """Load a YOLO or RT-DETR detection model."""
    t = model_type.lower()
    if t == 'yolo':
        return YOLO(model_path)
    elif t == 'rtdetr':
        return RTDETR(model_path)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose: 'yolo', 'rtdetr'")


# ------------------------------------------------------------
#  NMS helper
# ------------------------------------------------------------

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Non-Maximum Suppression over a pool of boxes.

    Args:
        boxes         : (N, 4) float array of [x1, y1, x2, y2].
        scores        : (N,)   float array of confidence scores.
        iou_threshold : Boxes whose IoU with a higher-scoring box exceeds this
                        value are suppressed. Lower = stricter.

    Returns:
        Integer indices of surviving boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)


# ------------------------------------------------------------
#  Auto-versioning
# ------------------------------------------------------------

def next_version(base_dir: Path) -> str:
    """Return the next v<N> folder name that does not yet exist under base_dir."""
    if not base_dir.exists():
        return "v1"
    nums = [
        int(p.name[1:])
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith('v') and p.name[1:].isdigit()
    ]
    return f"v{max(nums) + 1}" if nums else "v1"


# ------------------------------------------------------------
#  Main class
# ------------------------------------------------------------

class PersonTrackerCropper:
    """Detect (multi-model), filter, track (ReID), and crop persons from videos."""

    def __init__(
        self,
        # Detection models
        model_configs: list,
        # Output
        output_dir: str,
        csv_dir: str,
        # Tracker / ReID
        reid_weights: str = 'osnet_x0_25_msmt17.pt',
        tracker_name: str = 'botsort',
        half_precision: bool = False,
        # Tracker parameters
        det_thresh: float = 0.6,
        max_age: int = 60,
        min_hits: int = 3,
        # Filtering
        nms_iou_threshold: float = 0.5,
        min_area_px: int = 1000,
        min_hw_ratio: float = 1.5,
        # Processing
        frame_interval: int = 35,
        crop_padding: int = 10,
    ):
        """
        Args:
            model_configs     : List of dicts, each with:
                                  'path' (str)   - path to weights file
                                  'type' (str)   - 'yolo' or 'rtdetr'
                                  'conf' (float) - confidence threshold
            output_dir        : Base directory for crops (version sub-folder auto-created).
            csv_dir           : Base directory for CSV files (version sub-folder auto-created).
            reid_weights      : Path to BoxMOT ReID model weights file.
            tracker_name      : BoxMOT tracker to use (default 'botsort'). Options:
                                  With ReID   - 'botsort', 'strongsort', 'deepocsort',
                                               'boosttrack', 'hybridsort'
                                  Motion-only - 'bytetrack', 'ocsort'
            half_precision    : Use FP16 for ReID model inference (default False).
                                True is faster on GPU; keep False when running on CPU.
                                Has no effect on motion-only trackers.
            det_thresh        : Tracker's minimum confidence threshold (default 0.6).
                                Only detections with conf >= det_thresh will be tracked.
            max_age           : Maximum frames to keep track alive without detections (default 60).
                                Higher values maintain IDs through longer occlusions.
            min_hits          : Minimum consecutive detections before track confirmation (default 3).
                                Higher values reduce false tracks but slower initialization.
            nms_iou_threshold : IoU threshold for cross-model NMS (default 0.5).
                                Higher -> less suppression. Lower -> fewer, cleaner boxes.
            min_area_px       : Discard boxes with pixel area < this value (default 1000).
            min_hw_ratio      : Discard boxes where height < ratio * width (default 1.5).
            frame_interval    : Sample every N-th frame (default 35).
            crop_padding      : Pixel padding around each saved crop (default 10).
        """
        self.model_configs     = model_configs
        self.nms_iou_threshold = nms_iou_threshold
        self.min_area_px       = min_area_px
        self.min_hw_ratio      = min_hw_ratio
        self.frame_interval    = frame_interval
        self.crop_padding      = crop_padding
        self.reid_weights      = Path(reid_weights)
        self.tracker_name      = tracker_name.lower()
        self.half_precision    = half_precision
        self.det_thresh        = det_thresh
        self.max_age           = max_age
        self.min_hits          = min_hits
        self.device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Validate tracker name early
        if self.tracker_name not in TRACKER_MAP:
            valid = ', '.join(sorted(TRACKER_MAP.keys()))
            raise ValueError(f"Unknown tracker '{tracker_name}'. Valid options: {valid}")

        # Auto-versioning
        base_crops = Path(output_dir)
        base_csv   = Path(csv_dir)
        version    = next_version(base_crops)
        self.version    = version
        self.output_dir = base_crops / version
        self.csv_dir    = base_csv   / version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Load detection models
        self.models = []
        for cfg in model_configs:
            print(f"  Loading {cfg['type'].upper()} | conf={cfg['conf']} | {cfg['path']}")
            m = load_model(cfg['path'], cfg['type'])
            self.models.append({'model': m, 'conf': cfg['conf'], 'type': cfg['type']})
        print(f"  {len(self.models)} detection model(s) loaded.\n")

        # Tracker initialised per video
        self.tracker = None

        # Summary
        tracker_type = "ReID" if self.tracker_name in TRACKERS_WITH_REID else "motion-only"
        print("=" * 60)
        print(f"  Version           : {version}  (auto-detected)")
        print(f"  Crops dir         : {self.output_dir}")
        print(f"  CSV dir           : {self.csv_dir}")
        print(f"  Tracker           : {tracker_name}  ({tracker_type})")
        if self.tracker_name in TRACKERS_WITH_REID:
            print(f"  ReID weights      : {reid_weights}")
            print(f"  Half precision    : {half_precision}")
        print(f"  Tracker det_thresh: {det_thresh}")
        print(f"  Tracker max_age   : {max_age}")
        print(f"  Tracker min_hits  : {min_hits}")
        print(f"  Frame interval    : every {frame_interval} frames")
        print(f"  NMS IoU threshold : {nms_iou_threshold}")
        print(f"  Min area (px sq.) : {min_area_px}")
        print(f"  Min H/W ratio     : {min_hw_ratio}")
        print("=" * 60 + "\n")

    # ----------------------------------------------------------
    #  Private helpers
    # ----------------------------------------------------------

    def _init_tracker(self):
        """Instantiate a fresh tracker for each video."""
        self.tracker = build_tracker(
            tracker_name=self.tracker_name,
            reid_weights=self.reid_weights,
            device=self.device,
            half=self.half_precision,
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
        )

    def _run_all_models(self, frame: np.ndarray) -> np.ndarray:
        """
        Run all detection models on frame and pool person detections.

        Returns:
            (N, 7) array of [x1, y1, x2, y2, conf, cls=0, model_idx].
            model_idx is the integer index into self.models / self.model_names.
        """
        all_dets = []
        for idx, entry in enumerate(self.models):
            results = entry['model'].predict(
                frame, save=False, conf=entry['conf'], verbose=False
            )[0].boxes.cpu()

            if results is None or len(results.xyxy) == 0:
                continue

            boxes = results.xyxy.numpy()
            confs = results.conf.numpy()
            clss  = results.cls.numpy()

            # Person class = 0
            mask = clss == 0
            boxes, confs, clss = boxes[mask], confs[mask], clss[mask]

            if len(boxes) > 0:
                model_col = np.full(len(boxes), idx, dtype=np.float32)
                all_dets.append(np.column_stack([boxes, confs, clss, model_col]))

        return np.vstack(all_dets) if all_dets else np.empty((0, 7))

    def _apply_filters(self, dets: np.ndarray):
        """NMS -> area filter -> aspect-ratio filter.
        
        Expects (N, 7) input [x1, y1, x2, y2, conf, cls, model_idx].
        The model_idx column (col 6) is carried through all filters unchanged.
        NMS uses only cols 0-3 (boxes) and col 4 (conf) so model_idx never
        influences suppression decisions.
        """
        if len(dets) == 0:
            return dets, 0, 0, 0

        # 1. NMS — pass only the first 6 cols; keep model_idx via index selection
        keep = apply_nms(dets[:, :4], dets[:, 4], self.nms_iou_threshold)
        dets = dets[keep]
        n_after_nms = len(dets)

        # 2. Minimum area
        if len(dets):
            x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            dets = dets[(x2 - x1) * (y2 - y1) >= self.min_area_px]
        n_after_area = len(dets)

        # 3. Aspect ratio (height >= ratio * width)
        if len(dets):
            x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            dets = dets[(y2 - y1) >= self.min_hw_ratio * (x2 - x1)]
        n_after_ratio = len(dets)

        return dets, n_after_nms, n_after_area, n_after_ratio

    @staticmethod
    def _xyxy_to_yolo(box, img_w, img_h):
        x1, y1, x2, y2 = box
        return (
            (x1 + x2) / 2.0 / img_w,
            (y1 + y2) / 2.0 / img_h,
            (x2 - x1) / img_w,
            (y2 - y1) / img_h,
        )

    def _crop(self, frame: np.ndarray, box) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        H, W = frame.shape[:2]
        x1 = max(0, x1 - self.crop_padding)
        y1 = max(0, y1 - self.crop_padding)
        x2 = min(W, x2 + self.crop_padding)
        y2 = min(H, y2 + self.crop_padding)
        return frame[y1:y2, x1:x2]

    # ----------------------------------------------------------
    #  Public interface
    # ----------------------------------------------------------

    def process_video(self, video_path, save_crops: bool = True):
        """Detect, filter, track, and crop persons in a single video."""
        video_path = Path(video_path)
        video_name = video_path.stem

        print(f"\n{'-' * 60}")
        print(f"  Video : {video_name}")
        print(f"{'-' * 60}")

        video_crops_dir = self.output_dir / video_name
        if save_crops:
            video_crops_dir.mkdir(parents=True, exist_ok=True)

        self._init_tracker()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  Could not open video.")
            return

        # Derive a clean display name for each model (stem of the weights file)
        model_names = [Path(cfg['path']).stem for cfg in self.model_configs]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        csv_rows     = []
        crop_count   = 0
        unique_ids   = set()
        frame_no     = 0

        n_raw = n_after_nms = n_after_area = n_after_ratio = 0
        # Per-model raw detection counts (before any filtering)
        per_model_raw = {name: 0 for name in model_names}

        pbar = tqdm(total=total_frames, desc=f"  {video_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % self.frame_interval == 0:
                img_h, img_w = frame.shape[:2]

                # Detection — returns (N, 7): [x1,y1,x2,y2, conf, cls, model_idx]
                raw_dets = self._run_all_models(frame)
                n_raw   += len(raw_dets)

                # Tally raw counts per model
                if len(raw_dets):
                    for idx, name in enumerate(model_names):
                        per_model_raw[name] += int(np.sum(raw_dets[:, 6] == idx))

                # Filtering — model_idx rides through in col 6
                filtered, n_nms, n_area, n_ratio = self._apply_filters(raw_dets)
                n_after_nms   += n_nms
                n_after_area  += n_area
                n_after_ratio += n_ratio

                # Build a lookup: original row index in filtered -> model name
                # We need this to match tracked outputs back to their source model.
                # BoxMOT returns track[:, 7] as the index into the input dets array.
                det_model_by_idx = {}
                if len(filtered):
                    for row_i, det_row in enumerate(filtered):
                        det_model_by_idx[row_i] = model_names[int(det_row[6])]

                # Tracker expects (N, 6): [x1,y1,x2,y2, conf, cls] — strip col 6
                tracker_input = filtered[:, :6] if len(filtered) > 0 else np.empty((0, 6))

                # Tracking
                if len(tracker_input) > 0:
                    tracks = self.tracker.update(tracker_input, frame)
                else:
                    tracks = np.empty((0, 8))

                # Save crops + CSV rows
                for track in tracks:
                    x1, y1, x2, y2, person_id, conf, cls = track[:7]
                    person_id = int(person_id)
                    unique_ids.add(person_id)

                    # track[7] is the index of the matched detection in tracker_input
                    det_idx = int(track[7]) if len(track) > 7 else -1
                    det_model = det_model_by_idx.get(det_idx, 'unknown')

                    cx, cy, w, h = self._xyxy_to_yolo([x1, y1, x2, y2], img_w, img_h)

                    csv_rows.append({
                        'frame_no'  : frame_no,
                        'person_id' : person_id,
                        'class'     : int(cls),
                        'x_center'  : round(cx, 6),
                        'y_center'  : round(cy, 6),
                        'width'     : round(w,  6),
                        'height'    : round(h,  6),
                        'confidence': round(float(conf), 4),
                        'det_model' : det_model,
                    })

                    if save_crops:
                        crop  = self._crop(frame, [x1, y1, x2, y2])
                        fname = f"{video_name}_frame{frame_no:06d}_id{person_id:04d}.jpg"
                        cv2.imwrite(str(video_crops_dir / fname), crop)
                        crop_count += 1

            frame_no += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Filter summary
        print(f"\n  Filter summary:")
        print(f"    Raw detections      : {n_raw}")
        for name in model_names:
            count = per_model_raw[name]
            print(f"      {name:<30}: {count}")
        print(f"    After NMS           : {n_after_nms}")
        print(f"    After area filter   : {n_after_area}")
        print(f"    After ratio filter  : {n_after_ratio}")

        if csv_rows:
            df = pd.DataFrame(csv_rows)
            # Ensure det_model is the last column for readability
            cols = [c for c in df.columns if c != 'det_model'] + ['det_model']
            df = df[cols]
            csv_path = self.csv_dir / f"{video_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  CSV saved  -> {csv_path}  ({len(df)} rows)")
            print(f"  Unique IDs : {len(unique_ids)}")
        else:
            print("  No persons passed all filters.")

        if save_crops:
            print(f"  Crops saved -> {video_crops_dir}  ({crop_count} files)")

    def process_directory(self, video_dir, save_crops: bool = True, video_extensions: list = None):
        """Process all videos in a directory."""
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        video_dir   = Path(video_dir)
        video_files = []
        for ext in video_extensions:
            video_files.extend(sorted(video_dir.glob(f"*{ext}")))

        if not video_files:
            print(f"No video files found in {video_dir}")
            return

        print(f"Found {len(video_files)} video(s) in {video_dir}\n")

        for vf in video_files:
            self.process_video(vf, save_crops=save_crops)

        print("\n" + "=" * 60)
        print("  ALL VIDEOS PROCESSED")
        print(f"  Crops -> {self.output_dir}")
        print(f"  CSVs  -> {self.csv_dir}")
        print("=" * 60)


# ------------------------------------------------------------
#  Entry point -- edit everything below here
# ------------------------------------------------------------

def main():

    # Detection models -- add / remove dicts freely
    model_configs = [
        {
            'path': r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Detection Models/yolo_person_c0m_yv11.pt",
            'type': 'yolo',
            'conf': 0.3,
        },
        {
            'path': r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Detection Models/yolov8_person.pt",
            'type': 'yolo',
            'conf': 0.15,
        },
        {
            'path': r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Detection Models/rtdetr-l.pt",
            'type': 'rtdetr',
            'conf': 0.15,
        },
        {
            'path': r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Detection Models/yolo12m.pt",
            'type': 'yolo',
            'conf': 0.25,
        },
        {
            'path': r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Detection Models/yolov8m.pt",
            'type': 'yolo',
            'conf': 0.25,
        }
    ]

    # Paths
    video_directory  = r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/Original"
    output_crops_dir = r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/Person_Crops"
    output_csv_dir   = r"/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/Haram collected vids/YOLO_Coordinates"

    # Tracker settings
    # With ReID   : 'botsort' | 'strongsort' | 'deepocsort' | 'boosttrack' | 'hybridsort'
    # Motion-only : 'bytetrack' | 'ocsort'
    tracker_name   = 'boosttrack'
    reid_weights   = r'/home/mohammed/Desktop/Mohammed/UPM/Term 8/AI 492 - Capstone/ReID Models/clip_market1501.pt'   # auto-downloaded if not present
    half_precision = False                       # True for faster GPU inference

    # Tracker parameters
    det_thresh     = 0.35    # Tracker confidence threshold (0-1)
    max_age        = 90    # Frames to keep track alive without detections
    min_hits       = 3      # Consecutive detections before track confirmation

    # Processing
    frame_interval = 1
    crop_padding   = 1

    # Filtering
    nms_iou_threshold = 0.5   # NMS overlap threshold (0-1)
    min_area_px       = 600  # minimum bounding-box area in pixels
    min_hw_ratio      = 1.2   # height must be >= ratio * width

    # Run
    cropper = PersonTrackerCropper(
        model_configs     = model_configs,
        output_dir        = output_crops_dir,
        csv_dir           = output_csv_dir,
        reid_weights      = reid_weights,
        tracker_name      = tracker_name,
        half_precision    = half_precision,
        det_thresh        = det_thresh,
        max_age           = max_age,
        min_hits          = min_hits,
        nms_iou_threshold = nms_iou_threshold,
        min_area_px       = min_area_px,
        min_hw_ratio      = min_hw_ratio,
        frame_interval    = frame_interval,
        crop_padding      = crop_padding,
    )

    cropper.process_directory(
        video_dir  = video_directory,
        save_crops = True,
    )


if __name__ == "__main__":
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"Torch version  : {torch.__version__}\n")
    main()
