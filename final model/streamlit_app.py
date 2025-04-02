import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile
import os
from model import HybridModel  # Make sure this matches your model architecture

# Set up the app
st.set_page_config(page_title="Computer Parts Detector", layout="wide")
st.title("Computer Parts Detection System")
st.write("Upload an image to detect and classify computer components")

# Load model function with caching
@st.cache_resource
def load_model():
    # Initialize model (adjust parameters to match your model)
    model = HybridModel(num_classes=8)
    
    # Load trained weights (update path as needed)
    model_path = "final model/model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        st.error("Model weights not found! Please ensure 'best_model.pth' exists.")
        return None
    return model

# Define class names (update to match your dataset)
CLASS_NAMES = [
    "Computer-Parts", "CPU", "GPU", "HDD", 
    "MOTHERBOARD", "PSU", "RAM", "SSD"
]

# Define transforms (should match your training transforms)
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to process image and make predictions
def predict(image, model):
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert BGR to RGB if needed
    if img_np.shape[2] == 4:  # RGBA image
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    elif img_np.shape[2] == 1:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:  # Assume BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_transform()
    img_tensor = transform(img_np).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        class_output, detection_output = model(img_tensor)
    
    # Process classification output
    class_probs = torch.softmax(class_output, dim=1)
    top_prob, top_class = torch.max(class_probs, 1)
    
    # Process detection output
    boxes = detection_output[0]['boxes'].cpu().numpy()
    scores = detection_output[0]['scores'].cpu().numpy()
    labels = detection_output[0]['labels'].cpu().numpy()
    
    # Filter detections by confidence
    confidence_threshold = 0.5
    keep = scores > confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return {
        'classification': {
            'class_name': CLASS_NAMES[top_class.item()],
            'confidence': top_prob.item()
        },
        'detections': {
            'boxes': boxes,
            'scores': scores,
            'labels': [CLASS_NAMES[l] for l in labels]
        },
        'original_image': img_np
    }

# Function to draw bounding boxes
def draw_boxes(image, boxes, labels, scores):
    img = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label with class and confidence
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# Main app interface
def main():
    model = load_model()
    if model is None:
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        # Image upload options
        option = st.radio("Select input type:", 
                         ("Upload Image", "Use Webcam"))
        
        img_file = None
        
        if option == "Upload Image":
            img_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"])
        else:
            img_file = st.camera_input("Take a picture")
    
    with col2:
        st.header("Results")
        
        if img_file is not None:
            # Display original image
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            result = predict(np.array(image), model)
            
            # Show classification results
            st.subheader("Classification Results")
            st.write(f"Predicted Class: **{result['classification']['class_name']}**")
            st.write(f"Confidence: **{result['classification']['confidence']:.2%}**")
            
            # Show detection results
            st.subheader("Object Detection Results")
            if len(result['detections']['boxes']) > 0:
                # Draw boxes on image
                annotated_image = draw_boxes(
                    result['original_image'],
                    result['detections']['boxes'],
                    result['detections']['labels'],
                    result['detections']['scores']
                )
                st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                
                # Display detection details
                st.write("Detected Objects:")
                for box, label, score in zip(result['detections']['boxes'], 
                                           result['detections']['labels'], 
                                           result['detections']['scores']):
                    st.write(f"- {label} (confidence: {score:.2f})")
            else:
                st.warning("No computer parts detected in the image.")
            
            # Add download button for annotated image
            if len(result['detections']['boxes']) > 0:
                annotated_pil = Image.fromarray(annotated_image)
                with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
                    annotated_pil.save(tmp.name)
                    with open(tmp.name, "rb") as file:
                        st.download_button(
                            label="Download Annotated Image",
                            data=file,
                            file_name="detected_objects.jpg",
                            mime="image/jpeg"
                        )
        else:
            st.info("Please upload an image or use the webcam to test the model.")

if __name__ == "__main__":
    main()