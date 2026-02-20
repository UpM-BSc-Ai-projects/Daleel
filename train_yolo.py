from ultralytics import YOLO
import torch

MAP_THRESHOLD = 0.496

# --- Callback: stop training early if mAP50-95 surpasses the threshold ---
def early_stop_on_threshold(trainer):
    """Stop training and keep best.pt once mAP50-95 exceeds MAP_THRESHOLD."""
    map50_95 = trainer.metrics.get('metrics/mAP50-95(B)', 0.0)
    if map50_95 > MAP_THRESHOLD:
        epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        print(f"\n🎯 mAP50-95 = {map50_95:.4f} exceeded threshold {MAP_THRESHOLD} at epoch {epoch}.")
        print("   Stopping training early — best.pt has been saved.")
        raise StopIteration  # Ultralytics catches this and exits the training loop cleanly

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolo_person_c0m_yv11.pt')  # load a pretrained model (recommended for training)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    # Register the early-stop callback before training
    model.add_callback('on_fit_epoch_end', early_stop_on_threshold)

    # Train the model
    results = model.train(
        data=r'C:\Users\themi\PycharmProjects\Capstone2\Detection Fine-tune\data.yaml', 
        epochs=100, 
        imgsz=640, 
        device=device,
        batch=16,
        name='haram_yolo11m3',
        save=True
    )
