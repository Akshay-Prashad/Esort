import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
import os
import json
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Path to images folder.
            annotation_file (str): Path to COCO JSON annotations.
            transform (callable, optional): Image transformations.
        """
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(set(ann["image_id"] for ann in self.annotations["annotations"]))
        self.image_info = {img["id"]: img["file_name"] for img in self.annotations["images"]}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_info[image_id])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes & labels
        boxes, labels = [], []
        for ann in self.annotations["annotations"]:
            if ann["image_id"] == image_id:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
                labels.append(ann["category_id"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, targets

    def __getitem__(self, idx):
      image_id = self.image_ids[idx]
      image_path = os.path.join(self.image_dir, self.image_info[image_id])

      # Check if the image file exists
      if not os.path.exists(image_path):
          raise FileNotFoundError(f"Image file not found: {image_path}")

      image = cv2.imread(image_path)
      if image is None:
          raise ValueError(f"Failed to load image: {image_path}")

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Get bounding boxes & labels
      boxes, labels = [], []
      for ann in self.annotations["annotations"]:
          if ann["image_id"] == image_id:
              x, y, w, h = ann["bbox"]
              boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
              labels.append(ann["category_id"])

      # Convert to tensors
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      labels = torch.as_tensor(labels, dtype=torch.int64)

      targets = {"boxes": boxes, "labels": labels}

      if self.transform:
          image = self.transform(image)

      return image, targets

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


