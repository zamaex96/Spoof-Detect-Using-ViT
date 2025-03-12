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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Dataset path and model configuration
dataset_root = r"C:\SpoofDetect\dataset"

# Save the plots and models
output_folder = r"C:\SpoofDetect\ML\Models"  # Folder to save the model
os.makedirs(output_folder, exist_ok=True)
output_folder2 = r"C:\SpoofDetect\ML\CSV"  # Folder to save the CSV file
os.makedirs(output_folder2, exist_ok=True)
output_folder3 = r"C:\ML\Plots"  # Folder to save the plots
os.makedirs(output_folder3, exist_ok=True)

# Change model name to ViT-L/16
model_name = "vit_large_patch16_224"  # Vision Transformer Large model
ext = "SpoofDetect"
no_classes = 2
image_size = 224  # ViT expects an input size of 224x224
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

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
    plateau_scheduler.step(avg_val_loss)
    val_accuracy = total_val_correct / total_val_samples * 100
    validation_acc_arr.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

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

# Save loss and accuracy data to CSV
df = pd.DataFrame({'train_loss': training_loss_arr, 'val_loss': validation_loss_arr,
                   'train_acc': training_acc_arr, 'val_acc': validation_acc_arr})
csv_file_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
df.to_csv(csv_file_path, index=False)

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
