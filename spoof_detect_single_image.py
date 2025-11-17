
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from timm import create_model
from datetime import datetime
import argparse

# steps to execute
# Install all the requirements such as torch, PIL, os, pandas, timm etc
# 1. Downlaod the models from the repository
# 2. Insert the correct .pth file address of the model
# 3. Insert the folder path address where you have saved the image(s)
# 4. Run


# ==================== USER CONFIGURATION ====================
# CHANGE THIS PATH TO YOUR TEST IMAGES FOLDER
IMAGE_FOLDER_PATH = r"C:\SpoofDetect\test_images"   # <-- UPDATE THIS PATH ACCORDINGLY

# Model & settings (keep as-is)
MODEL_PATH = r"C:\SpoofDetect\ML\Models\vit_large_patch16_224_spoofdetect.pth" # <-- UPDATE THIS PATH ACCORDINGLY
MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 2
IMAGE_SIZE = 224
CLASS_NAMES = ["Real", "Fake"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output CSV (auto-saved next to model)
OUTPUT_CSV = os.path.join(
    os.path.dirname(MODEL_PATH),
    f"spoofdetect_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
# ============================================================

# ------------------- PRE-PROCESSING -------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ------------------- LOAD MODEL -------------------
def load_model():
    print("[INFO] Loading model...")
    model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('model.', '').replace('module.', '').replace('backbone.', '')
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print(f"[OK] Model loaded: {MODEL_PATH}")
    return model

# ------------------- PREDICT SINGLE IMAGE -------------------
@torch.no_grad()
def predict_image(model, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())
        pred_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

        return {
            "filename": os.path.basename(img_path),
            "prediction": pred_label,
            "confidence_%": round(confidence, 2),
            "prob_real_%": round(probs[0] * 100, 2),
            "prob_fake_%": round(probs[1] * 100, 2)
        }
    except Exception as e:
        return {
            "filename": os.path.basename(img_path),
            "prediction": "ERROR",
            "confidence_%": 0,
            "prob_real_%": 0,
            "prob_fake_%": 0,
            "error": str(e)
        }

# ------------------- PROCESS FOLDER -------------------
def process_folder():
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"[ERROR] Folder not found: {IMAGE_FOLDER_PATH}")
        return

    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    img_files = [os.path.join(IMAGE_FOLDER_PATH, f) for f in os.listdir(IMAGE_FOLDER_PATH)
                 if f.lower().endswith(supported)]

    if not img_files:
        print(f"[WARN] No images found in: {IMAGE_FOLDER_PATH}")
        return

    print(f"[INFO] Found {len(img_files)} images in:")
    print(f"     {IMAGE_FOLDER_PATH}")
    print(f"[INFO] Starting inference...\n")

    model = load_model()
    results = []

    for fp in img_files:
        res = predict_image(model, fp)
        results.append(res)
        status = res["prediction"]
        conf = res["confidence_%"]
        print(f"{res['filename']:30} → {status:4} ({conf:6.2f}%)")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Results saved to:")
    print(f"     {OUTPUT_CSV}")

# ------------------- RUN -------------------
if __name__ == "__main__":
    print("="*60)
    print("SPOOF IMAGE DETECTOR (ViT-Large)")
    print("="*60)
    process_folder()
