# Vision Transformer (ViT) Implementation for Spoofing Detection

This comprehensive code implements a deep learning solution for spoofing detection using a Vision Transformer (ViT) architecture. The system leverages transfer learning by fine-tuning a pre-trained ViT model for binary classification, distinguishing between genuine and spoofed images in a specialized security dataset.

## Purpose and Goal

The primary purpose of this implementation is to create a robust image classification system that can accurately detect spoofing attempts in digital media. Spoofing detection is a critical security application where the model learns to distinguish between authentic images and manipulated or falsified ones. The code utilizes a state-of-the-art Vision Transformer architecture, which has demonstrated superior performance in complex visual recognition tasks compared to traditional convolutional networks.

## Implementation Steps

### Environment and Library Setup

The code begins by importing necessary libraries for deep learning (PyTorch), data handling (torchvision), visualization (matplotlib), and efficient model implementation (timm). The environment variable `KMP_DUPLICATE_LIB_OK` is set to manage potential library conflicts, ensuring smooth execution across different systems.

### Directory Configuration

The implementation establishes a structured file organization system with dedicated directories for:
- Input data (`C:\SpoofDetect\dataset`)
- Trained models (`C:\SpoofDetect\ML\Models`)
- Performance metrics in CSV format (`C:\SpoofDetect\ML\CSV`)
- Visualization plots (`C:\ML\Plots`)

Each directory is created if it doesn't already exist, ensuring a fail-safe execution environment.

### Model and Hyperparameter Configuration

The code specifies crucial configuration parameters for the model training process:
- Model architecture: `vit_large_patch16_224` (Vision Transformer Large with 16×16 pixel patches)
- Classification task: Binary classification (2 classes)
- Input dimensions: 224×224 pixels (standard for ViT models)
- Training duration: 50 epochs
- Mini-batch size: 64 samples
- Learning rate: 0.001 (suitable for fine-tuning pre-trained models)

Hardware acceleration is automatically configured based on the availability of CUDA-compatible GPUs.

### Model Architecture Definition

A custom `ImageClassifier` class extends PyTorch's neural network module to encapsulate the Vision Transformer model. This implementation:
- Loads the pre-trained ViT-Large model with 16×16 patch size using the `timm` library
- Replaces the final classification layer to accommodate the binary spoofing detection task
- Maintains the transfer learning benefits while adapting to the specific domain

### Data Processing Pipeline

The data processing pipeline incorporates several critical transformations:
- Resizing all images to 224×224 pixels to match ViT input requirements
- Converting images to PyTorch tensors
- Normalizing pixel values using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

The dataset is loaded from structured directories using torchvision's `ImageFolder`, which automatically assigns class labels based on folder organization. Separate data loaders are created for training and validation with appropriate batching and shuffling.

### Training and Validation Loop

The implementation follows a standard deep learning training paradigm over multiple epochs:

For each epoch, the code:
- Sets the model to training mode with gradient calculation enabled
- Processes each batch of training images through the model
- Calculates cross-entropy loss between predictions and ground truth
- Updates model weights through backpropagation
- Tracks training loss and accuracy metrics

After each training epoch, a validation phase:
- Sets the model to evaluation mode (disabling dropouts and batch normalization updates)
- Processes validation images without gradient calculation
- Computes validation loss and accuracy
- Records metrics for performance analysis

The comprehensive logging system outputs performance metrics after each epoch, providing real-time feedback on training progress.

### Model Preservation and Analysis

Upon completion of training, the implementation:
- Saves the trained model with full configuration details for future deployment
- Exports training and validation metrics to CSV format for detailed analysis
- Generates and saves visualization plots showing the progression of loss and accuracy
- Creates both high-resolution PNG and PDF formats of the visualizations

The visualization component includes dual plots displaying:
- Training and validation loss curves to assess convergence and potential overfitting
- Training and validation accuracy curves to evaluate classification performance

## Technical Significance

This implementation represents a sophisticated approach to spoofing detection using transformer-based deep learning. The code demonstrates several advanced practices:
- Transfer learning with state-of-the-art Vision Transformer architecture
- Comprehensive training and evaluation pipeline
- Detailed performance tracking and visualization
- Production-ready model saving with metadata

The binary classification focus suggests application in security-critical domains where distinguishing between authentic and spoofed images has significant implications for digital trust and security systems.

# Step-by-Step Implementation of Vision Transformer for Spoofing Detection

## 1. Environment and Library Setup

The first step is to import all necessary libraries for the implementation. This includes PyTorch for the deep learning framework, torchvision for dataset management, matplotlib for visualization, and timm (PyTorch Image Models) for accessing pre-trained Vision Transformer models.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import pandas as pd
from timm import create_model  # Using timm to load ViT
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fix for macOS library conflicts
```

The last line sets an environment variable to prevent potential OpenMP library conflicts, particularly on macOS systems, which could cause runtime errors.

## 2. Directory Configuration and Project Structure

Next, set up the directory structure for the project. This organizes the workflow and ensures that outputs are saved in appropriate locations.

```python
# Dataset path and model configuration
dataset_root = r"C:\SpoofDetect\dataset"

# Save the plots and models
output_folder = r"C:\SpoofDetect\ML\Models"  # Folder to save the model
os.makedirs(output_folder, exist_ok=True)
output_folder2 = r"C:\SpoofDetect\ML\CSV"  # Folder to save the CSV file
os.makedirs(output_folder2, exist_ok=True)
output_folder3 = r"C:\ML\Plots"  # Folder to save the plots
os.makedirs(output_folder3, exist_ok=True)
```

The `os.makedirs()` function with `exist_ok=True` creates these directories if they don't already exist, preventing errors during execution. This design follows a clean separation of concerns, with dedicated locations for models, performance metrics, and visualizations.

## 3. Model and Hyperparameter Configuration

Define the model architecture and set hyperparameters that control the training process. These values significantly influence the model's performance and training efficiency.

```python
# Change model name to ViT-L/16
model_name = "vit_large_patch16_224"  # Vision Transformer Large model
ext = "SpoofDetect"
no_classes = 2  # Binary classification for spoof detection
image_size = 224  # ViT expects an input size of 224x224
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

The code automatically selects GPU acceleration if available, which significantly speeds up training. The hyperparameters are chosen based on empirical knowledge for transfer learning scenarios with Vision Transformers.

## 4. Model Architecture Definition

Create a custom neural network class that incorporates the pre-trained Vision Transformer and adapts it for the specific spoofing detection task.

```python
# Model Definition
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, model_name):
        super(ImageClassifier, self).__init__()
        if model_name == "vit_large_patch16_224":
            # Using timm to load the Vision Transformer (ViT-L/16) with pre-trained weights
            self.model = create_model(model_name, pretrained=True)
            # Adjust the classifier to match the number of classes
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = ImageClassifier(no_classes, model_name).to(device)
```

This class loads the pre-trained ViT-L/16 model using the timm library and replaces the final classification layer to output predictions for the binary spoofing detection task. The `to(device)` method moves the model to the appropriate computing device (GPU or CPU).

## 5. Optimization and Loss Function Configuration

Define the optimization algorithm and loss function for training the model. These components drive the learning process.

```python
# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
```

The Adam optimizer is selected for its adaptive learning rate properties, making it suitable for fine-tuning pre-trained models. Cross-entropy loss is the standard choice for classification problems, measuring the difference between predicted class probabilities and actual class labels.

## 6. Data Transformation and Loading

Prepare the dataset with appropriate transformations and create data loaders for efficient batch processing during training and validation.

```python
# Define transforms (ViT requires image input size of 224x224)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=transform)
validation_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Training and Validation arrays
training_loss_arr = []
validation_loss_arr = []
training_acc_arr = []
validation_acc_arr = []
```

The transformations resize images to 224×224 pixels (required by ViT), convert them to tensors, and normalize them using ImageNet statistics. The `ImageFolder` class automatically assigns class labels based on the subdirectory structure of the dataset. Data loaders enable efficient batch processing with optional shuffling for better training dynamics.

## 7. Training and Validation Loop Implementation

Implement the core training loop that iterates through epochs, processes batches, and updates the model weights. This also includes validation to monitor performance on unseen data.

```python
for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train_correct += (predicted == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_loss_arr.append(avg_train_loss)

    train_accuracy = total_train_correct / total_train_samples * 100
    training_acc_arr.append(train_accuracy)

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val_correct += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss = total_val_loss / len(validation_dataloader)
    validation_loss_arr.append(avg_val_loss)

    val_accuracy = total_val_correct / total_val_samples * 100
    validation_acc_arr.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
```

This section contains several key components:
- Setting the model to training mode with `model.train()` activates dropout and batch normalization
- The gradient reset, forward pass, loss calculation, backward pass, and optimizer step form the core training loop
- Accuracy calculation helps monitor performance in human-interpretable terms
- Setting the model to evaluation mode with `model.eval()` and using `torch.no_grad()` ensures validation is performed without updating weights or consuming unnecessary memory
- Comprehensive logging provides real-time feedback on training progress

## 8. Model Saving and Serialization

After training completes, save the trained model along with relevant metadata for future use.

```python
# Save the final model
model_path = os.path.join(output_folder, f"{model_name}_{ext}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'image_size': image_size,
    'no_classes': no_classes,
    'num_epochs': num_epochs,
    'hyperparameters': {
        'image_size': image_size,
        'no_classes': no_classes,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
    }
}, model_path)
print(f"Model saved at {model_path}")
```

This code saves the model's learned parameters along with configuration details. Storing these hyperparameters with the model enables easier reproduction and deployment in the future. The naming convention incorporates both the model architecture and the specific application.

## 9. Performance Metrics Recording

Save training and validation metrics to a CSV file for detailed analysis and future reference.

```python
# Save loss and accuracy data to CSV
df = pd.DataFrame({'train_loss': training_loss_arr, 'val_loss': validation_loss_arr,
                   'train_acc': training_acc_arr, 'val_acc': validation_acc_arr})
csv_file_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
df.to_csv(csv_file_path, index=False)
```

Using pandas for data management provides a clean interface for storing performance metrics. The resulting CSV file can be used for further analysis, comparison with other models, or visualization in external tools.

## 10. Visualization and Analysis

Create visual representations of the training process to analyze model performance and convergence behavior.

```python
# Plot loss and accuracy curves
plt.figure(figsize=(12, 6), dpi=300)

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(training_loss_arr, label="Training Loss", color='b')
plt.plot(validation_loss_arr, label="Validation Loss", color='r')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
plt.title('Loss Curve', fontsize=14, fontweight='bold')
plt.legend()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(training_acc_arr, label="Training Accuracy", color='b')
plt.plot(validation_acc_arr, label="Validation Accuracy", color='r')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
plt.legend()

# Save the plots
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
plt.show()
```

The visualization creates a two-panel figure showing:
- Loss curves to indicate convergence and potential overfitting (if validation loss increases while training loss decreases)
- Accuracy curves to demonstrate classification performance
- Both plots are saved in high-resolution PNG and PDF formats for inclusion in reports or presentations

These visualizations are critical for understanding model behavior and making informed decisions about potential improvements to the training process.

## Implementation Considerations

For optimal results when implementing this code:

1. **Dataset Organization**: Ensure your dataset is organized with "train" and "test" directories, each containing subdirectories for each class (e.g., "real" and "spoof")

2. **Hardware Requirements**: Vision Transformers are computationally intensive. While the code will run on CPU, GPU acceleration is strongly recommended for practical training times

3. **Hyperparameter Tuning**: The provided values (batch size, learning rate, etc.) are reasonable starting points, but optimal values may depend on your specific dataset and hardware

4. **Model Size Considerations**: The ViT-Large model has approximately 307M parameters. If computational resources are limited, consider using "vit_base_patch16_224" (86M parameters) instead

5. **Early Stopping**: For production environments, consider implementing early stopping based on validation metrics to prevent overfitting and reduce training time

This implementation provides a complete pipeline for training a state-of-the-art Vision Transformer model for spoofing detection, with comprehensive logging, visualization, and model preservation for future use.

# Vision Transformer (ViT) Confusion Matrix Evaluation for Spoofing Detection

This code implements a comprehensive evaluation pipeline for a pre-trained Vision Transformer model used in spoofing detection. The primary goal is to assess the model's performance in distinguishing between real and fake (spoofed) images, generating detailed performance metrics, and creating visualizations to interpret the results.

## Purpose and Goal

The purpose of this implementation is to thoroughly evaluate a previously trained Vision Transformer (ViT) model for spoofing detection. The code loads a trained model, applies it to a test dataset, and generates comprehensive performance analytics. This evaluation framework helps in understanding the effectiveness of the spoofing detection system by measuring its accuracy, precision, recall, and F1 score, while also providing visual interpretations through confusion matrices.

## Step-by-Step Implementation

### 1. Library Imports and Environment Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from torchvision.models import vit_large_patch16_224
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
```

The implementation begins by importing necessary libraries for deep learning (PyTorch), data handling, metrics calculation (scikit-learn), visualization (matplotlib and seaborn), and time-based file naming. This comprehensive set of imports provides all the functionality needed for model evaluation and results presentation.

### 2. Configuration and Path Setup

```python
# Dataset path and model configuration
dataset_root = r"C:\SpoofDetect\dataset"
model_path = r"C:\SpoofDetect\ML\Models\vit_large_patch16_224_spoofdetect.pth"
model_name = "vit_large_patch16_224"
ext = "conf_mat_spoof_detect"
no_classes = 2
image_size = 224  # ViT expects 224x224 input
batch_size = 64
class_names = ["Real", "Fake"]
output_folder1 = r"C:\SpoofDetect\ML\Plots"
os.makedirs(output_folder1, exist_ok=True)
```

This section defines the configuration parameters for the evaluation process, including file paths, model specifications, and output directories. The code ensures all output directories exist before proceeding, preventing runtime errors during file saving operations.

### 3. Confusion Matrix Visualization Function

```python
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
    
    # Styling and labels
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Percentage', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    cbar.ax.tick_params(labelsize=16 + delta_font_size + fixed_size)
    
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', 
              fontsize=20 + delta_font_size + fixed_size, fontweight='bold')
    plt.xlabel('Predicted', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    plt.ylabel('True', fontsize=18 + delta_font_size + fixed_size, fontweight='bold')
    
    # Save and display
    png_file_path = os.path.join(output_folder1, f"{model_name}_{ext}.png")
    plt.savefig(png_file_path, format='png', dpi=1500)
    plt.show()
```

This function creates a highly customized confusion matrix visualization using seaborn's heatmap. It normalizes the confusion matrix to show percentages rather than raw counts, making it easier to interpret class-wise performance. The visualization includes careful attention to styling details like font sizes, weights, and color schemes for professional-quality output.

### 4. Reproducibility Configuration

```python
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

To ensure reproducible results across different runs, this section sets fixed random seeds for all libraries that involve randomness (PyTorch, NumPy, Python's random). For GPU operations, additional settings ensure deterministic behavior, though at a potential cost to performance.

### 5. Device Setup and Model Initialization

```python
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the Vision Transformer model
model_inference = vit_large_patch16_224(weights=None)
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
```

This section initializes the Vision Transformer model architecture and loads the pre-trained weights. The code is designed to be robust to different saved model formats by handling various dictionary structures. It also cleans up the state dictionary keys to ensure compatibility, which is important when loading models trained with different code versions or distributed training setups.

### 6. Data Transformation and Loading

```python
# Define transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
```

This section prepares the test data for evaluation. It defines a transformation pipeline that includes resizing to the required 224×224 pixels for ViT models, as well as various data augmentations. Interestingly, data augmentation is applied during evaluation, which is unusual but can help assess the model's robustness to variations in the input. The code also reports the dataset size and class distribution, which is crucial for interpreting evaluation results.

### 7. Model Evaluation

```python
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
```

The core evaluation loop processes the test dataset in batches, running each batch through the model and collecting predictions. Using `torch.no_grad()` ensures no gradients are computed, saving memory and computation time during inference. The implementation collects both true and predicted labels for further analysis while also calculating and displaying the overall accuracy.

### 8. Performance Metrics Calculation

```python
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
```

This section calculates comprehensive performance metrics beyond simple accuracy. It computes per-class accuracy from the confusion matrix, which is crucial for imbalanced datasets where high overall accuracy might hide poor performance on minority classes. It also calculates precision, recall, and F1 score using macro-averaging, which gives equal weight to all classes regardless of their frequency.

### 9. Metrics Saving

```python
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
```

This comprehensive reporting section saves all evaluation results to a text file. The file name includes a timestamp to prevent overwriting previous results. The report includes model configuration details, overall performance metrics, and per-class accuracy statistics, creating a complete record of the evaluation process.

### 10. Visualization Generation

```python
# Plot confusion matrix
plot_confusion_matrix(true_labels, predicted_labels, class_names)
```

Finally, the code calls the previously defined function to generate, display, and save the confusion matrix visualization. This provides a visual representation of the model's performance, making it easier to identify patterns such as which classes are most frequently confused with each other.

## Summary

This evaluation framework provides a thorough assessment of a Vision Transformer model for spoofing detection. By calculating diverse metrics and generating visualizations, it offers insights into model performance beyond simple accuracy. The detailed documentation and flexible code structure make it adaptable for evaluating other image classification models with minimal modifications.

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="clickMe.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>


