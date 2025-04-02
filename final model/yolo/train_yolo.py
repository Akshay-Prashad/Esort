import os
import yaml
import shutil
from pycocotools.coco import COCO
from yolo_model import ComputerPartsYOLO
def prepare_yolo_dataset(dataset_path):
    """Convert COCO to YOLO format with proper image paths"""
    yolo_dir = os.path.abspath("_yolo_dataset")

    # Clean and recreate directory
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
    os.makedirs(yolo_dir)

    for phase in ['train', 'val']:
        # Create directories
        phase_dir = os.path.join(yolo_dir, phase)
        images_dir = os.path.join(phase_dir, 'images')
        labels_dir = os.path.join(phase_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Copy images (not symlinks)
        src_images = os.path.join(dataset_path, 'train/' if phase == 'train' else 'valid/')
        for img_file in os.listdir(src_images):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(
                    os.path.join(src_images, img_file),
                    os.path.join(images_dir, img_file)
                )

        # Convert COCO annotations
        coco = COCO(os.path.join(dataset_path, 'train/' if phase == 'train' else 'valid/', '_annotations.coco.json'))

        for img_id in coco.getImgIds():
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Create label file
            base_name = os.path.splitext(img_info['file_name'])[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            with open(label_path, 'w') as f:
                for ann in anns:
                    # Convert COCO bbox to YOLO format
                    x, y, w, h = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']

                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    width = w / img_w
                    height = h / img_h

                    f.write(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Create dataset.yaml
    data = {
        'train': os.path.join(yolo_dir, "train"),
        'val': os.path.join(yolo_dir, "val"),
        'names': ['Computer-Parts','CPU', 'GPU', 'HDD', 'MOTHERBOARD', 'PSU', 'RAM', 'SSD', ],
        'nc': 8
    }

    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Dataset prepared at: {yaml_path}")
    print(f"Train images: {len(os.listdir(os.path.join(yolo_dir, 'train')))}")
    print(f"Val images: {len(os.listdir(os.path.join(yolo_dir, 'val')))}")
    return yaml_path

def verify_dataset(yaml_path):
    """Verify the dataset is properly formatted"""
    import cv2
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    for phase in ['train', 'val']:
        phase_dir = data[phase]
        images_dir = os.path.join(phase_dir, 'images')
        labels_dir = os.path.join(phase_dir, 'labels')

        print(f"\nVerifying {phase}:")
        print(f"Images: {len(os.listdir(images_dir))}")
        print(f"Labels: {len(os.listdir(labels_dir))}")

        # Check one sample
        img_file = os.listdir(images_dir)[0]
        label_file = os.path.splitext(img_file)[0] + '.txt'

        img = cv2.imread(os.path.join(images_dir, img_file))
        h, w = img.shape[:2]

        print(f"Sample {img_file} ({w}x{h}):")
        with open(os.path.join(labels_dir, label_file)) as f:
            print(f.read())

def train():
    # Configuration
    DATASET_PATH = "drive/MyDrive/Dataset"
    MODEL_SIZE = "n"
    EPOCHS = 10
    BATCH_SIZE = 8
    IMG_SIZE = 640
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare dataset
    print("Preparing dataset...")
    data_yaml = prepare_yolo_dataset(DATASET_PATH)
    verify_dataset(data_yaml)

    # Train model
    print("\nStarting training...")
    model = ComputerPartsYOLO(num_classes=8, model_size=MODEL_SIZE)
    model.train(
        data_yaml=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    model.save("best_yolov8_model.pt")
    print("Training completed!")

if __name__ == "__main__":
    import torch
    train()