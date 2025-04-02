import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()

        # Feature extractor
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove classification head

        # Detection model
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace box predictor head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Classification head
        self.fc = nn.Linear(1280, num_classes)  # EfficientNet-B0 features

    def forward(self, x, targets=None):
        """
        Args:
            x: Input images (4D tensor [B,C,H,W] or list of 3D tensors [C,H,W])
            targets: Optional list of target dictionaries
        """
        # Convert input to proper format
        if isinstance(x, list):
            # List of 3D tensors -> stack to 4D
            x = torch.stack(x)
            batch_mode = True
        elif x.dim() == 3:
            # Single 3D tensor -> add batch dim
            x = x.unsqueeze(0)
            batch_mode = False
        elif x.dim() == 4:
            batch_mode = True
        else:
            raise ValueError(f"Expected input dim 3 or 4, got {x.dim()}")

        # Feature extraction for classification
        features = self.efficientnet(x)  # [B, 1280]
        class_output = self.fc(features)  # [B, num_classes]

        # Detection branch - requires list of images
        if batch_mode:
            # Convert batch to list of images
            img_list = [img for img in x]
        else:
            img_list = [x.squeeze(0)]

        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided in training mode")
            detection_output = self.detector(img_list, targets)
        else:
            detection_output = self.detector(img_list)

        return class_output, detection_output

    def predict(self, x):
        """Inference-only forward pass"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

