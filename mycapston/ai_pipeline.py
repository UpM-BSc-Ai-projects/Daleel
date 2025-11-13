# Requirements: pip install 

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from transformers import pipeline

print("Loading AI Pipeline Helpers...")

# --- Copied from your original script ---

def get_special_model_transform(input_size=224):
    """
    Returns the *exact* transforms for your MobileNet model.
    !!! VERIFY THIS matches your training script !!!
    """
    return T.Compose([
        T.Resize(int(input_size * 1.14)), # 256 for 224 input
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_special_model(model_path, device, num_classes=2):
    """
    Loads your fine-tuned MobileNetV3-Large model.
    """
    print(f"Loading special model from: {model_path}")
    model = models.mobilenet_v3_large(weights=None) 
    try:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    except Exception as e:
        print(f"!!! Error modifying MobileNetV3-Large classifier: {e}")
        raise e
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"!!! Error loading model weights from {model_path}: {e}")
        raise e
        
    model.to(device)
    model.eval()
    print("Special model loaded successfully.")
    return model

# --- New Functions for This Workflow ---

def load_all_pipelines(special_model_path, device_num=0):
    """
    Loads all three AI models once and returns them in a dictionary.
    """
    print("Loading all AI models. This may take a moment...")
    
    # 1. Set device
    torch_device = torch.device(f"cuda:{device_num}" if device_num >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    # 2. Load Emotion model
    emotion_clf = pipeline("image-classification", 
                           model="trpakov/vit-face-expression", 
                           device=device_num)
    
    # 3. Load Age model
    age_clf = pipeline("image-classification", 
                       model="nateraw/vit-age-classifier", 
                       device=device_num)
    
    # 4. Load Special model
    special_model = load_special_model(special_model_path, 
                                       torch_device, 
                                       num_classes=2) # ["normal", "special"]
    special_transform = get_special_model_transform()
    
    print("All models loaded.")
    return {
        "emotion_clf": emotion_clf,
        "age_clf": age_clf,
        "special_model": special_model,
        "special_transform": special_transform,
        "device": torch_device
    }

def run_ai_on_crop(crop_image, models, special_model_labels=["normal", "special"]):
    """
    Runs all three models on a *single* cropped PIL Image.
    Returns a dictionary with labels and scores.
    """
    
    results = {
        "emotion": {"label": "unknown", "score": 0.0},
        "age": {"label": "unknown", "score": 0.0},
        "special": {"label": "unknown", "score": 0.0}
    }

    # 1. Run Emotion
    try:
        pred_e = models["emotion_clf"](crop_image, top_k=1)
        if pred_e:
            results["emotion"] = pred_e[0]
    except Exception as e:
        print(f"Error in emotion model: {e}")

    # 2. Run Age
    try:
        pred_a = models["age_clf"](crop_image, top_k=1)
        if pred_a:
            results["age"] = pred_a[0]
    except Exception as e:
        print(f"Error in age model: {e}")
        
    # 3. Run Special Model
    try:
        device = models["device"]
        model = models["special_model"]
        transform = models["special_transform"]
        
        # Transform the image and add a batch dimension
        transformed_img = transform(crop_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(transformed_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, index = torch.max(probabilities, 1)
            
            label_idx = index.cpu().item()
            results["special"] = {
                "label": special_model_labels[label_idx],
                "score": score.cpu().item()
            }
    except Exception as e:
        print(f"Error in special model: {e}")

    return results