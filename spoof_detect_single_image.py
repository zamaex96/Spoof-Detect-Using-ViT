
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from timm import create_model
from datetime import datetime
import argparse

# ------------------- CONFIGURATION -------------------
MODEL_PATH = r"C:\SpoofDetect\ML\Models\vit_large_patch16_224_spoofdetect.pth"
MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 2
IMAGE_SIZE = 224
CLASS_NAMES = ["Real", "Fake"]          # index 0 = Real, 1 = Fake
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- PRE-PROCESSING -------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ------------------- LOAD MODEL -------------------
def load_model():
    model = create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    # Replace head (just in case the checkpoint does not contain it)
    if not hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
    else:
        state_dict = ckpt

    # Strip common prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("model.", "module.", "backbone."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print(f"[OK] Model loaded from {MODEL_PATH}")
    return model

# ------------------- SINGLE IMAGE INFERENCE -------------------
@torch.no_grad()
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, C, H, W)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(logits.argmax(dim=1).item())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    return pred_label, confidence, probs

# ------------------- BATCH PROCESS FOLDER -------------------
def process_folder(model, folder_path, output_csv=None):
    results = []
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    img_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(supported)]

    if not img_files:
        print(f"[WARN] No images found in {folder_path}")
        return

    print(f"[INFO] Found {len(img_files)} images → starting inference")
    for fp in img_files:
        try:
            label, conf, prob_vec = predict_image(model, fp)
            results.append({
                "filename": os.path.basename(fp),
                "prediction": label,
                "confidence_%": round(conf, 2),
                "prob_real_%": round(prob_vec[0] * 100, 2),
                "prob_fake_%": round(prob_vec[1] * 100, 2)
            })
            print(f"{os.path.basename(fp)} → {label} ({conf:.2f}%)")
        except Exception as e:
            print(f"[ERROR] {fp} → {e}")

    # ---- Save CSV report (optional) ----
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"[OK] Report saved → {output_csv}")

    return results

# ------------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run spoof-detection on every image in a folder")
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Path to folder containing images to test")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional CSV file to store results")
    args = parser.parse_args()

    model = load_model()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out or os.path.join(
        os.path.dirname(MODEL_PATH),
        f"spoofdetect_inference_{timestamp}.csv")

    process_folder(model, args.folder, csv_path)
