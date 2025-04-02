# from ultralytics import YOLO
# import torch
# import os

# class ComputerPartsYOLO:
#     def __init__(self, num_classes=8, model_size='n'):
#         """Initialize YOLOv8 model"""
#         self.model = YOLO(f'yolov8{model_size}.pt')
#         self.model.model.nc = num_classes  # Update number of classes

#     def train(self, data_yaml, epochs=100, imgsz=640, batch=8, device='cpu'):
#         """Train the model with absolute paths"""
#         # Convert to absolute path
#         data_yaml = os.path.abspath(data_yaml)

#         self.model.train(
#             data=data_yaml,
#             epochs=epochs,
#             imgsz=imgsz,
#             batch=batch,
#             device=device,
#             pretrained=True,
#             optimizer='AdamW',
#             lr0=1e-4,
#             augment=True,
#             exist_ok=True  # Overwrite existing runs
#         )

#     def save(self, path):
#         """Save model weights"""
#         torch.save(self.model.model.state_dict(), path)
from ultralytics import YOLO
import torch
import os

class ComputerPartsYOLO:
    def __init__(self, model_path=None, num_classes=8):
        """Initialize with proper error handling"""
        self.model = None
        self.model_path = model_path
        self.num_classes = num_classes
        
        if model_path and os.path.exists(model_path):
            self._safe_load_model()
        
    def _safe_load_model(self):
        """Safely load model with proper initialization"""
        try:
            self.model = YOLO(self.model_path)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'nc'):
                self.model.model.nc = self.num_classes
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            self.model = None
            return False
    
    def __call__(self, *args, **kwargs):
        """Make model callable with safety checks"""
        if self.model is None and not self._safe_load_model():
            raise RuntimeError("Model failed to load")
        return self.model(*args, **kwargs)
    
    def predict(self, image, conf=0.25):
        """Prediction with proper error handling"""
        try:
            return self(image, conf=conf)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None