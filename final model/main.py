import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocessor import CustomDataset, transform
from model import HybridModel

# Hyperparameters
BATCH_SIZE = 4
NUM_CLASSES = 8
EPOCHS = 1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train_dataset = CustomDataset(image_dir="Dataset/valid", annotation_file="Dataset/valid/_annotations.coco.json", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load model
model = HybridModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
classification_loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)

    for images, targets in progress_bar:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        class_output, detection_output = model(torch.stack(images), targets)

        # Compute losses
        loss_class = classification_loss_fn(
            class_output,
            torch.tensor([t["labels"][0] for t in targets], dtype=torch.long).to(DEVICE),
        )
        loss_detection = detection_output["loss_classifier"] + detection_output["loss_box_reg"] + \
                         detection_output["loss_objectness"] + detection_output["loss_rpn_box_reg"]

        loss = loss_class + loss_detection
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {epoch_loss / len(train_loader):.4f}")

print("Training Complete.")
