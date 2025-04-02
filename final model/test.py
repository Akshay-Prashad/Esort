import torch
from model import HybridModel
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(num_classes=8).to(device)
model.load_state_dict(torch.load("final model/model.pth", map_location=device))
model.eval()

# Test image path (change to your image)
TEST_IMAGE = "final model/Dataset/valid/Seagate-1TB-Desktop-HDD-Internal-Hard-Disk-Drive-7200-RPM-SATA-6Gb-s-64MB-Cache-34_jpg.rf.7c7d0efcbcd1ff8638b2bfa5477c902d.jpg"
# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and process image
image = cv2.imread(TEST_IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_image = image.copy()
image_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    class_output, detections = model(image_tensor)

# Process classification
class_probs = torch.softmax(class_output, dim=1)
top_prob, top_class = torch.max(class_probs, 1)
class_names = ["background","CPU", "GPU", "HDD", "MOTHERBOARD", "PSU", "RAM", "SSD"]

# Process detections
if isinstance(detections, list) and len(detections) > 0:
    boxes = detections[0]['boxes'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()
else:
    boxes, scores, labels = np.array([]), np.array([]), np.array([])

# Scale boxes to original image size
h, w = image.shape[:2]
scale_x = w / 320
scale_y = h / 320
boxes[:, [0, 2]] *= scale_x
boxes[:, [1, 3]] *= scale_y

# Draw results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
result_image = original_image.copy()
for box, score, label in zip(boxes, scores, labels):
    if score > 0.1:  # Confidence threshold
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(result_image, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(result_image)
plt.title(f"Classification: {class_names[top_class.item()]} ({top_prob.item():.2%})\nDetected: {len(boxes)} components")
plt.show()

print("\n=== Debug Information ===")
print(f"Classification: {class_names[top_class.item()]} ({top_prob.item():.2%})")
print(f"Detection output type: {type(detections)}")
if isinstance(detections, list):
    print(f"Number of detection sets: {len(detections)}")
    if len(detections) > 0:
        print("First detection keys:", detections[0].keys())
        print(f"Number of boxes: {len(detections[0]['boxes'])}")
elif isinstance(detections, dict):
    print("Detection keys:", detections.keys())
print("Final boxes shape:", boxes.shape)