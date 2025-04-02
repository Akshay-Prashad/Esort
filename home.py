import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import time
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="PC Parts Management Suite",
    layout="wide",
    page_icon="🖥️",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'purchased_parts' not in st.session_state:
    st.session_state.purchased_parts = []
if 'detected_parts' not in st.session_state:
    st.session_state.detected_parts = []

# Load models (cached)
@st.cache_resource
def load_detection_model():
    try:
        return YOLO("saved_models/best_yolov8_model.pt")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Home Page
def home_page():
    st.title("🖥️ PC Parts Management Suite")
    st.markdown("""
    ## Your Complete PC Component Solution
    
    Navigate between our three powerful modules using the links below:
    """)
    
    # Page links in a nice layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.page_link("pages/app1.py", label="**Component Detection**", icon="🔍")
        st.markdown("""
        Identify computer parts using AI-powered image recognition.
        Upload images to detect and inventory your components.
        """)
    
    with col2:
        st.page_link("pages/app2.py", label="**Compatibility Advisor**", icon="🔄")
        st.markdown("""
        Get smart recommendations for compatible parts.
        Build your perfect system with confidence.
        """)
    
    with col3:
        st.page_link("pages/app3.py", label="**Economic Analysis**", icon="💰")
        st.markdown("""
        Evaluate resale vs recycling options.
        Make data-driven decisions about your components.
        """)
    
    st.markdown("---")
    
    # Current parts inventory
    st.subheader("📦 Your Current Parts Inventory")
    if st.session_state.purchased_parts:
        for part in st.session_state.purchased_parts:
            st.write(f"- {part}")
        
        if st.button("Clear All Parts", type="primary"):
            st.session_state.purchased_parts = []
            st.rerun()
    else:
        st.info("No parts added yet. Start by detecting components!")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Components", len(st.session_state.purchased_parts))
    
    with col2:
        if st.session_state.purchased_parts:
            unique_types = len(set(st.session_state.purchased_parts))
            st.metric("Unique Types", unique_types)
        else:
            st.metric("Unique Types", 0)

# Component Detection Page
def detection_page():
    st.title("🔍 Computer Parts Detection")
    st.markdown("Upload an image to detect computer components using YOLOv8")
    
    model = load_detection_model()
    if model is None:
        return
    
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
    
    # Sidebar with navigation
    with st.sidebar:
        st.page_link("Home.py", label="← Back to Home", icon="🏠")
        st.header("Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.25
        )
    
    # Main content
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with st.spinner("Detecting computer parts..."):
            img_np = np.array(image)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            
            results = model(img_np, conf=conf_threshold)
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy().astype(int)
            
            result_img = img_np.copy()
            detected_parts = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                label_text = f"{CLASS_NAMES[label]}: {score:.2f}"
                cv2.putText(result_img, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected_parts.append(CLASS_NAMES[label])
            
            st.session_state.detected_parts = list(set(detected_parts))
            
            with col2:
                st.image(result_img, caption="Detection Results", use_column_width=True)
            
            st.subheader("Detected Components")
            if st.session_state.detected_parts:
                for part in st.session_state.detected_parts:
                    if st.button(f"Add {part} to my parts list"):
                        if part not in st.session_state.purchased_parts:
                            st.session_state.purchased_parts.append(part)
                            st.success(f"Added {part} to your parts list!")
                            time.sleep(1)
                            st.rerun()
            else:
                st.warning("No parts detected above confidence threshold")

# App routing (simplified for example)
# In a real app, you would split these into separate files as shown in the page_link targets
current_page = "Home"  # This would be determined by your routing logic

if current_page == "Home":
    home_page()
elif current_page == "Component Detection":
    detection_page()
# Other pages would follow the same pattern