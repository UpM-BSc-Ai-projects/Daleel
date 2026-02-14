import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, RTDETR
from tqdm import tqdm
import time
import os

class ModelEvaluator:
    def __init__(self, gt_path, video_path, output_file='model_evaluation_results.csv'):
        """
        Initialize the evaluator.
        
        Args:
            gt_path (str): Path to the Ground Truth CSV file.
            video_path (str): Path to the video file.
            output_file (str): Path to save the final evaluation results CSV.
        """
        self.gt_path = gt_path
        self.video_path = video_path
        self.output_file = output_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Ground Truth
        print(f"Loading Ground Truth from: {gt_path}")
        self.gt_df = pd.read_csv(gt_path)
        
        # Normalize GT columns if needed or just ensure expected format
        # Expected GT columns: frame_no, class, x_center, y_center, width, height, (confidence ignored for GT)
        required_cols = ['frame_no', 'x_center', 'y_center', 'width', 'height']
        if not all(col in self.gt_df.columns for col in required_cols):
            raise ValueError(f"Ground Truth CSV missing required columns: {required_cols}")
            
        print(f"Ground Truth loaded. {len(self.gt_df)} annotations found.")

    def load_model(self, model_path):
        """Load a model (YOLO or RTDETR) based on file extension or content."""
        print(f"Loading model: {model_path}")
        # Ultralytics handles both YOLO and RTDETR via the YOLO class or specific RTDETR class
        # But usually YOLO() class can load rtdetr .pt files too, or we can use RTDETR() explicitly 
        # if the filename indicates it.
        
        path_str = str(model_path).lower()
        if 'rtdetr' in path_str:
            try:
                model = RTDETR(model_path)
            except Exception:
                # Fallback to YOLO class if RTDETR fails (sometimes weights are saved differently)
                model = YOLO(model_path)
        else:
            model = YOLO(model_path)
            
        return model

    def get_iou(self, box1, box2):
        """
        Calculate IoU between two boxes (x_center, y_center, w, h).
        Normalized or pixels, as long as both are same scale.
        """
        # box: [xc, yc, w, h]
        b1_x1 = box1[0] - box1[2] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_y2 = box1[1] + box1[3] / 2
        
        b2_x1 = box2[0] - box2[2] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_y2 = box2[1] + box2[3] / 2
        
        # Intersection
        xi1 = max(b1_x1, b2_x1)
        yi1 = max(b1_y1, b2_y1)
        xi2 = min(b1_x2, b2_x2)
        yi2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        b1_area = box1[2] * box1[3]
        b2_area = box2[2] * box2[3]
        
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def calculate_ap(self, recalls, precisions):
        """Calculate Average Precision using 11-point interpolation or area under curve."""
        # Simple area under curve approximation
        # Append sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        method = 'continuous' 
        if method == 'continuous':
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        else:
            # 11-point (VOC2007)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0
                
        return ap

    def evaluate_single_model(self, model_path, conf_threshold=0.25, frame_interval=35):
        """
        Evaluate a single model on the video.
        """
        model = self.load_model(model_path)
        model_name = Path(model_path).name
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        predictions = []
        
        # Speed measurement
        inference_times = []
        
        print(f"Running inference for {model_name}...")
        # Estimate total processed frames for progress bar
        processed_frames_total = (total_frames + frame_interval - 1) // frame_interval
        pbar = tqdm(total=processed_frames_total, desc=f"Inferencing {model_name}")
        
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on interval
            if frame_no % frame_interval != 0:
                frame_no += 1
                continue
            
            # Run Inference
            start_time = time.time()
            results = model.predict(frame, conf=conf_threshold, verbose=False, device=self.device)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000) # ms
            
            # Parse results
            # We need normalized xywh to match GT
            h, w = frame.shape[:2]
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # check class, we only want persons (class 0 for COCO)
                    cls = int(box.cls[0])
                    if cls != 0: 
                        continue
                        
                    conf = float(box.conf[0])
                    # box.xywhn is normalized xywh
                    xywhn = box.xywhn[0].tolist()
                    
                    predictions.append({
                        'frame_no': frame_no,
                        'x_center': xywhn[0],
                        'y_center': xywhn[1],
                        'width': xywhn[2],
                        'height': xywhn[3],
                        'confidence': conf
                    })
            
            pbar.update(1)
            frame_no += 1
            
        cap.release()
        pbar.close()
        
        if not predictions:
            print(f"No predictions made by {model_name}!")
            return None

        pred_df = pd.DataFrame(predictions)
        
        # Calculate Metrics
        avg_speed_ms = np.mean(inference_times)
        fps = 1000.0 / avg_speed_ms
        
        metrics = {
            'Model': model_name,
            'Inference_Speed_ms': avg_speed_ms,
            'FPS': fps
        }
        
        # Calculate mAP and other metrics for different IoU thresholds
        # First, filter GT to only frames that were processed (in case video ended early or GT is partial)
        valid_frames = set(pred_df['frame_no'].unique()).union(set(self.gt_df['frame_no'].unique()))
        
        gt_filtered = self.gt_df[self.gt_df['frame_no'].isin(valid_frames)]
        
        # Calculate AP for each threshold from 0.50 to 0.95 step 0.05
        iou_thresholds = np.arange(0.50, 1.00, 0.05)
        ap_accum = 0.0
        
        for iou_thresh in iou_thresholds:
            # Round iou_thresh to 2 decimal places to avoid floating point issues
            iou_thresh = round(iou_thresh, 2)
            result = self.calculate_metrics_at_iou(pred_df, gt_filtered, iou_thresh)
            
            ap_accum += result['ap']
            
            # Store specific thresholds
            if iou_thresh == 0.50:
                metrics['mAP_50'] = result['ap']
                metrics['Precision'] = result['precision']
                metrics['Recall'] = result['recall']
                metrics['F1_Score'] = result['f1']
                metrics['TP'] = result['tp']
                metrics['FP'] = result['fp']
                metrics['FN'] = result['fn']
            elif iou_thresh == 0.75:
                metrics['mAP_75'] = result['ap']
            elif iou_thresh == 0.95:
                metrics['mAP_95'] = result['ap']

        # Calculate mAP_50-95
        metrics['mAP_50-95'] = ap_accum / len(iou_thresholds)
                
        return metrics

    def calculate_metrics_at_iou(self, pred_df, gt_df, iou_threshold):
        """Calculate AP, P, R, F1, TP, FP, FN at a specific IoU threshold."""
        
        # Sort predictions by confidence desc
        pred_df = pred_df.sort_values(by='confidence', ascending=False)
        
        tp_list = np.zeros(len(pred_df))
        fp_list = np.zeros(len(pred_df))
        
        gt_by_frame = gt_df.groupby('frame_no')
        
        # Track which GT boxes have been matched
        # dict: frame -> [bool, bool, ...]
        gt_matched = {}
        total_gt = len(gt_df)
        
        # Pre-process GT for faster lookup
        gt_lookup = {}
        for frame, group in gt_by_frame:
            boxes = group[['x_center', 'y_center', 'width', 'height']].values.tolist()
            gt_lookup[frame] = {
                'boxes': boxes,
                'matched': [False] * len(boxes)
            }
            
        # Iterate predictions
        for i, pred in enumerate(pred_df.itertuples()):
            frame = pred.frame_no
            pred_box = [pred.x_center, pred.y_center, pred.width, pred.height]
            
            if frame not in gt_lookup:
                # Prediction in a frame with no GT -> False Positive
                fp_list[i] = 1
                continue
                
            frame_gt = gt_lookup[frame]
            gt_boxes = frame_gt['boxes']
            
            best_iou = -1.0
            best_idx = -1
            
            # Find best overlapping GT
            for idx, gt_box in enumerate(gt_boxes):
                iou = self.get_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold:
                if not frame_gt['matched'][best_idx]:
                    tp_list[i] = 1
                    frame_gt['matched'][best_idx] = True
                else:
                    fp_list[i] = 1 # Duplicate detection
            else:
                fp_list[i] = 1
                
        # Compute cumulative metrics
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)
        recalls = tp_cumsum / total_gt
        
        ap = self.calculate_ap(recalls, precisions)
        
        # "Spot" metrics (at the end of the list means at the lowest confidence considered)
        # However, typically people want the metrics at the point that maximizes F1, or just the final count.
        # Let's use the final cumulative counts for TP/FP/FN (which corresponds to all preds > conf_threshold)
        # And P/R/F1 from that.
        
        final_tp = np.sum(tp_list)
        final_fp = np.sum(fp_list)
        final_fn = total_gt - final_tp
        
        if final_tp + final_fp > 0:
            p_score = final_tp / (final_tp + final_fp)
        else:
            p_score = 0.0
            
        if total_gt > 0:
            r_score = final_tp / total_gt
        else:
            r_score = 0.0
            
        if p_score + r_score > 0:
            f1_score = 2 * p_score * r_score / (p_score + r_score)
        else:
            f1_score = 0.0
            
        return {
            'ap': ap,
            'precision': p_score,
            'recall': r_score,
            'f1': f1_score,
            'tp': int(final_tp),
            'fp': int(final_fp),
            'fn': int(final_fn)
        }

    def run(self, model_paths, frame_interval=35):
        results = []
        
        for model_path in model_paths:
            print(f"\n{'='*50}")
            print(f"Evaluating Model: {model_path}")
            print(f"{'='*50}")
            
            try:
                metrics = self.evaluate_single_model(model_path, frame_interval=frame_interval)
                if metrics:
                    results.append(metrics)
                    print("\nResults:")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            print(f"{k}: {v:.4f}")
                        else:
                            print(f"{k}: {v}")
            except Exception as e:
                print(f"Error evaluating {model_path}: {e}")
                import traceback
                traceback.print_exc()
                
        return results

# ==========================================
# CONFIGURATION AND MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Base Paths
    # IMPORTANT: Ensure these directories exist and contain files named Vid_9.mp4, Vid_9.csv, etc.
    video_base_dir = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\Haram_Videos"
    gt_base_dir = r"C:\Users\themi\PycharmProjects\Capstone2\Datasets\person_crops_v3\yolo_coords"
    
    output_file = 'model_evaluation_results_all_videos.csv'
    
    # 2. Frame Processing Interval
    frame_interval = 35
    
    # 3. List of models to evaluate
    models_to_evaluate = [
        # Detected models in the directory:
        r"C:\Users\themi\PycharmProjects\Capstone2\Tracking Evaluator\rtdetr-l.pt",
        r"C:\Users\themi\PycharmProjects\Capstone2\Tracking Evaluator\yolov8m.pt",
        r"C:\Users\themi\PycharmProjects\Capstone2\Tracking Evaluator\yolov8_person.pt",
        r"C:\Users\themi\PycharmProjects\Capstone2\Tracking Evaluator\yolo12m.pt",
        r"C:\Users\themi\PycharmProjects\Capstone2\Tracking Evaluator\yolo_person_c0m_yv11.pt"
    ]
    
    # Check if files exist
    valid_models = []
    for m in models_to_evaluate:
        if Path(m).exists():
            valid_models.append(m)
        else:
            # Maybe it's a standard ultralytics model name that will be downloaded
            if not Path(m).is_absolute() and len(Path(m).parts) == 1:
                valid_models.append(m)
            else:
                print(f"Warning: Model file not found: {m}")
    
    if not valid_models:
        print("No valid models found to evaluate.")
    else:
        all_results = []
        
        # Iterate videos 9 to 13
        for i in range(9, 14):
            video_filename = f"Vid_{i}.mp4"
            gt_filename = f"Vid_{i}.csv"
            
            video_path = os.path.join(video_base_dir, video_filename)
            gt_path = os.path.join(gt_base_dir, gt_filename)
            
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            if not os.path.exists(gt_path):
                print(f"Ground truth not found: {gt_path}")
                continue
                
            print(f"\nProcessing Video {i}: {video_path}")
            
            try:
                # Instantiate evaluator for each video
                evaluator = ModelEvaluator(gt_path, video_path)
                video_results = evaluator.run(valid_models, frame_interval=frame_interval)
                
                # Append video identifier to results
                for res in video_results:
                    res['Video'] = f"Vid_{i}"
                    all_results.append(res)
            except Exception as e:
                print(f"Error processing video {i}: {e}")
                import traceback
                traceback.print_exc()
                
        # Save all results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            # Reorder columns
            cols = ['Video', 'Model', 'mAP_50', 'mAP_75', 'mAP_95', 'mAP_50-95', 'Precision', 'Recall', 'F1_Score', 
                    'TP', 'FP', 'FN', 'Inference_Speed_ms', 'FPS']
            # Add any extra columns that might exist
            existing_cols = [c for c in cols if c in df.columns]
            # Add remaining columns
            remaining = [c for c in df.columns if c not in existing_cols]
            df = df[existing_cols + remaining]
            
            df.to_csv(output_file, index=False)
            print(f"\nFinal results saved to {output_file}")
            print(df)
        else:
            print("No results to save.")
