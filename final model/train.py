import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocessor import CustomDataset, transform
from model import HybridModel
from torch.optim.lr_scheduler import StepLR
import os

# Hyperparameters
BATCH_SIZE = 16
NUM_CLASSES = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "best_model.pth"
SAVE_DIR = "saved_models"  # Directory to save all models
os.makedirs(SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Load datasets
train_dataset = CustomDataset(
    image_dir="/content/drive/MyDrive/Dataset/train",
    annotation_file="/content/drive/MyDrive/Dataset/train/_annotations.coco.json",
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_dataset = CustomDataset(
    image_dir="/content/drive/MyDrive/Dataset/valid",
    annotation_file="/content/drive/MyDrive/Dataset/valid/_annotations.coco.json",
    transform=transform
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# Load model
model = HybridModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
classification_loss_fn = torch.nn.CrossEntropyLoss()

# Initialize best metrics
best_val_accuracy = 0.0
best_val_loss = float('inf')

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)

    for images, targets in progress_bar:
        # Move data to device
        images = [torch.tensor(img, dtype=torch.float32).to(DEVICE) for img in images]
        targets = [{
            k: torch.tensor(v).to(DEVICE) if isinstance(v, torch.Tensor)
            else torch.tensor(v, dtype=torch.long).to(DEVICE)
            for k, v in t.items()
        } for t in targets]

        optimizer.zero_grad()

        # Forward pass
        class_output, detection_output = model(torch.stack(images), targets)

        # Compute losses
        loss_class = classification_loss_fn(
            class_output,
            torch.tensor([t["labels"][0] for t in targets], dtype=torch.long).to(DEVICE),
        )
        loss_detection = sum([v for k, v in detection_output.items() if k.startswith('loss')])
        loss = loss_class + loss_detection

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(class_output, 1)
        total += len(targets)
        correct += (predicted == torch.tensor([t["labels"][0] for t in targets], dtype=torch.long).to(DEVICE)).sum().item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{100 * correct / total:.2f}%")

    # Update learning rate
    scheduler.step()

    # Calculate epoch metrics
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

    # Validation
    #

    # Save model if either accuracy improves or loss decreases
    save_model = False

    if epoch_accuracy > best_val_accuracy:
        best_val_accuracy = epoch_accuracy
        print(f"New best validation accuracy: {best_val_accuracy:.2f}%")
        save_model = True

    if avg_epoch_loss < best_val_loss:
        best_val_loss = avg_epoch_loss
        print(f"New best validation loss: {best_val_loss:.4f}")
        save_model = True

    if save_model:
        # Save current model
        model_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}_acc_{epoch_accuracy:.2f}_loss_{avg_epoch_loss:.4f}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

        # Update best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model updated at epoch {epoch+1}")

print("Training Complete.")
