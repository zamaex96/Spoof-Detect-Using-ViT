import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# Dataset path and model configuration
dataset_root = r"C:\SpoofDetect\dataset"
model_path = r"C:\SpoofDetect\ML\Models\vit_large_patch16_224_spoofdetect.pth"  # Updated model path
model_name = "vit_large_patch16_224"  # Updated model name
ext = "conf_mat_spoof_detect"  # Updated extension for ViT
no_classes = 2
image_size = 224  # ViT expects 224x224 input
batch_size = 64
class_names = ["Real", "Fake"]
output_folder1 = r"C:\SpoofDetect\ML\Plots"
os.makedirs(output_folder1, exist_ok=True)
delta_font_size = -5
fixed_size = 5

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    accuracy = accuracy_score(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='cubehelix',
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=0.8, linecolor='black',
                     cbar_kws={'label': 'Percentage'},
                     annot_kws={"size": 18 + delta_font_size, "weight": "bold"})
    # Increase colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Percentage', fontsize=18 + delta_font_size + fixed_size)
    # Increase colorbar tick label size
    cbar.ax.tick_params(labelsize=16 + delta_font_size + fixed_size)
    cbar.ax.set_ylabel('Percentage', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')

    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', fontsize=20 + delta_font_size + fixed_size, fontweight='bold')
    plt.xlabel('Predicted', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    plt.ylabel('True', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    plt.xticks(fontsize=16 + delta_font_size + fixed_size, rotation=0, fontweight='bold')
    plt.yticks(fontsize=16 + delta_font_size + fixed_size, rotation=0, fontweight='bold')
    png_file_path = os.path.join(output_folder1, f"{model_name}_{ext}.png")
    plt.savefig(png_file_path, format='png', dpi=1500)
    plt.show()

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the Vision Transformer model with the correct number of classes
model_inference = vit_large_patch16_224(weights=None)  # Load ViT without pretrained weights
# Modify the head layer to match your number of classes
model_inference.heads.head = nn.Linear(model_inference.heads.head.in_features, no_classes)

# Load the trained weights
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Clean up state dict keys
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('model.', '').replace('module.', '')
    new_state_dict[new_key] = v

# Load the state dict
model_inference.load_state_dict(new_state_dict, strict=False)
model_inference = model_inference.to(device)
model_inference.eval()

# Define transforms (ViT uses similar normalization to ResNet)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ViT uses ImageNet normalization
])

# Load validation dataset
validation_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Print dataset information
print(f"\nValidation set size: {len(validation_dataset)} images")
print("Class distribution:")
for idx, class_name in enumerate(validation_dataset.classes):
    class_count = sum(1 for _, label in validation_dataset if label == idx)
    print(f"Class {class_name}: {class_count} images")

# Evaluate model
true_labels = []
predicted_labels = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in validation_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_inference(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Print per-class accuracy
print("\nPer-class accuracy:")
cm = confusion_matrix(true_labels, predicted_labels)
for i, class_name in enumerate(validation_dataset.classes):
    class_correct = cm[i, i]
    class_total = cm[i].sum()
    class_acc = class_correct / class_total * 100
    print(f"{class_name}: {class_acc:.2f}%")

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels, average='macro') * 100
recall = recall_score(true_labels, predicted_labels, average='macro') * 100
f1 = f1_score(true_labels, predicted_labels, average='macro') * 100

# Print metrics
print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# Save metrics to text file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder2 = r"C:\SpoofDetect\ML\Metrics"
os.makedirs(output_folder2, exist_ok=True)
metrics_filename = os.path.join(output_folder2, f"model_metrics_{model_name}_{ext}_{timestamp}.txt")

with open(metrics_filename, 'w') as f:
    # Save configuration details
    f.write("Model Configuration:\n")
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Extension: {ext}\n")
    f.write(f"Number of Classes: {no_classes}\n")
    f.write(f"Image Size: {image_size}x{image_size}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Class Names: {', '.join(class_names)}\n\n")

    # Save performance metrics
    f.write("Performance Metrics:\n")
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    f.write(f"Precision (macro): {precision:.2f}%\n")
    f.write(f"Recall (macro): {recall:.2f}%\n")
    f.write(f"F1 Score (macro): {f1:.2f}%\n\n")

    # Detailed per-class accuracy
    f.write("Per-class Accuracy:\n")
    for i, class_name in enumerate(validation_dataset.classes):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_acc = class_correct / class_total * 100
        f.write(f"{class_name}: {class_acc:.2f}%\n")

print(f"\nMetrics saved to {metrics_filename}")

# Plot confusion matrix
plot_confusion_matrix(true_labels, predicted_labels, class_names)
