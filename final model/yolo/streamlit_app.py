import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Configuration
MODEL_PATH = "saved_models/best_yolov8_model.pt"  # Your trained model
CLASS_NAMES = {
    0: 'Computer-Parts',
    1: 'CPU',
    2: 'GPU',
    3: 'HDD',
    4: 'MOTHERBOARD',
    5: 'PSU',
    6: 'RAM',
    7: 'SSD'
}

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

def predict_image(model, image, conf_threshold):
    """Run prediction on an image"""
    # Convert PIL to numpy array
    img_np = np.array(image)
    
    # Handle different image formats
    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:  # RGBA
        img_np = img_np[:, :, :3]
    
    # Run inference (matches your training config)
    results = model(
        img_np,
        conf=conf_threshold,
        iou=0.7,
        imgsz=640,
        augment=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy().astype(int)
    
    return img_np, boxes, scores, labels

def draw_detections(image, boxes, scores, labels):
    """Draw bounding boxes and labels"""
    result_img = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
        cv2.putText(result_img, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result_img

def main():
    st.set_page_config(page_title="Computer Parts Detector", layout="wide")
    st.title("🔍 Computer Parts Detection (YOLOv8)")
    
    # Sidebar controls (matches your training config)
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25,  # Default from your args.yaml
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IOU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,  # From your args.yaml
        help="Intersection Over Union threshold"
    )
    
    # Load model
    model = load_model()
    
    # Input options
    input_method = st.radio("Input Method:", ["Upload Image", "Webcam Capture"])
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            with st.spinner("🔎 Detecting computer parts..."):
                img_np, boxes, scores, labels = predict_image(model, image, conf_threshold)
                
                if len(boxes) > 0:
                    result_img = draw_detections(img_np, boxes, scores, labels)
                    st.image(result_img, caption="Detection Results", use_column_width=True)
                    
                    # Detection details
                    st.subheader("📋 Detection Details")
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels), 1):
                        st.success(
                            f"{i}. **{CLASS_NAMES[label]}**  \n"
                            f"Confidence: {score:.2f}  \n"
                            f"Bounding Box: {box.astype(int)}"
                        )
                else:
                    st.warning("⚠️ No computer parts detected above confidence threshold")
    
    else:  # Webcam
        st.warning("Webcam requires browser permissions")
        picture = st.camera_input("Take a picture of computer parts")
        
        if picture:
            image = Image.open(picture)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            with st.spinner("Processing..."):
                img_np, boxes, scores, labels = predict_image(model, image, conf_threshold)
                
                if len(boxes) > 0:
                    result_img = draw_detections(img_np, boxes, scores, labels)
                    st.image(result_img, caption="Webcam Detection", use_column_width=True)
                    
                    st.subheader("Detected Components")
                    for box, score, label in zip(boxes, scores, labels):
                        st.write(f"- {CLASS_NAMES[label]} (confidence: {score:.2f})")
                else:
                    st.warning("No parts detected in webcam image")

if __name__ == "__main__":
    import torch
    main()