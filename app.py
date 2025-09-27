import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time
import cv2

# Configure page
st.set_page_config(
    page_title="🐠 Fish Species Classifier",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model"""
    try:
        model_files = ['multicls_fish_model_yolo.pt', 'multiclassFish_mobilenetv2.keras']
        model = None
        
        for filename in model_files:
            try:
                model = YOLO(filename)
                st.session_state.model_filename = filename
                break
            except:
                continue
        
        if model is None:
            raise Exception("No YOLO model file found")
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please place your YOLO model file (multicls_fish_model_yolo.pt,multiclassFish_mobilenetv2.keras, etc.) in the same directory")
        return None

def get_model_info(model):
    """Get information about the YOLO model"""
    if model is None:
        return {}
    
    try:
        info = {
            'classes': list(model.names.values()),
            'num_classes': len(model.names),
            'model_type': 'YOLO',
        }
        return info
    except:
        return {'classes': [], 'num_classes': 0, 'model_type': 'YOLO'}

def predict_fish_species(image, model):
    """Make prediction using YOLO model"""
    if model is None:
        return []
    
    try:
        # Convert PIL image to numpy array for YOLO
        img_array = np.array(image)
        
        # Make prediction
        results = model(img_array, verbose=False)
        
        # Extract predictions
        predictions = []
        
        # Get the first result
        result = results[0]
        
        # Get class names from the model
        class_names = model.names
        
        if result.probs is not None:
            # Classification task - get probabilities
            probs = result.probs.data.cpu().numpy()
            
            # Create results list with class names and probabilities
            for i, prob in enumerate(probs):
                if i in class_names:
                    predictions.append((class_names[i], float(prob)))
            
            # Sort by probability (highest first)
            predictions.sort(key=lambda x: x[1], reverse=True)
            
        elif result.boxes is not None:
            # Detection task - get detected objects with confidence
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            
            # Group by class and take highest confidence for each class
            class_confidences = {}
            for class_id, conf in zip(class_ids, confidences):
                class_name = class_names[int(class_id)]
                if class_name not in class_confidences or conf > class_confidences[class_name]:
                    class_confidences[class_name] = conf
            
            # Convert to list and sort
            predictions = list(class_confidences.items())
            predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🐠 AI Fish Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced YOLO model for accurate fish species identification</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Information")
        model = load_model()
        if model is not None:
            st.success("✅ YOLO Model loaded successfully!")
            model_info = get_model_info(model)
            
            # Show model filename if available
            if hasattr(st.session_state, 'model_filename'):
                st.info(f"**File:** {st.session_state.model_filename}")
            
            st.info(f"**Type:** {model_info.get('model_type', 'YOLO')}")
            st.info(f"**Classes:** {model_info.get('num_classes', 'N/A')}")
            
            # Show detected classes
            if model_info.get('classes'):
                st.markdown("**Fish Species in Model:**")
                for i, class_name in enumerate(model_info['classes'][:10]):  # Show first 10
                    st.write(f"• {class_name}")
                if len(model_info['classes']) > 10:
                    st.write(f"... and {len(model_info['classes']) - 10} more")
        else:
            st.error("❌ Model not found!")
            st.warning("Place your YOLO model file in the app directory")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)
        show_top_n = st.selectbox("Show Top N Predictions", [3, 5, 10], index=0)
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        This YOLO model can identify fish species from images with high accuracy. 
        Upload a clear image of a fish to get started!
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Fish Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a fish for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"📏 Size: {image.size[0]} × {image.size[1]} pixels")
            st.write(f"🎨 Mode: {image.mode}")
            st.write(f"📁 Format: {image.format}")
    
    with col2:
        st.markdown("### 🔍 Classification Results")
        
        if uploaded_file is not None:
            # Add prediction button
            if st.button("🚀 Classify Fish Species", type="primary"):
                model = load_model()
                if model is None:
                    st.error("❌ Cannot make prediction: Model not loaded!")
                    st.info("Please ensure your YOLO model file is in the same directory as this script")
                else:
                    # Show loading animation
                    with st.spinner("🧠 AI is analyzing the image..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            time.sleep(0.01)
                    
                    # Make prediction
                    predictions = predict_fish_species(image, model)
                    
                    if predictions:
                        # Display top prediction
                        top_prediction = predictions[0]
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🎯 Predicted Species</h2>
                            <h1>{top_prediction[0]}</h1>
                            <h3>Confidence: {top_prediction[1]:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show confidence level
                        if top_prediction[1] >= confidence_threshold:
                            st.success(f"✅ High confidence prediction!")
                        else:
                            st.warning(f"⚠️ Low confidence. Consider uploading a clearer image.")
                        
                        # Filter predictions above threshold
                        filtered_predictions = [p for p in predictions if p[1] >= confidence_threshold]
                        if not filtered_predictions:
                            filtered_predictions = predictions[:show_top_n]
                        
                        # Display top N predictions
                        display_predictions = filtered_predictions[:show_top_n]
                        st.markdown(f"### 📊 Top {len(display_predictions)} Predictions")
                        
                        # Create visualization
                        species_names = [pred[0] for pred in display_predictions]
                        confidences = [pred[1] * 100 for pred in display_predictions]
                        
                        # Horizontal bar chart
                        fig = px.bar(
                            x=confidences,
                            y=species_names,
                            orientation='h',
                            title="Prediction Confidence Scores",
                            labels={'x': 'Confidence (%)', 'y': 'Fish Species'},
                            color=confidences,
                            color_continuous_scale='viridis'
                        )
                        
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Results")
                        results_df = pd.DataFrame(display_predictions, columns=['Species', 'Confidence'])
                        results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.1%}")
                        results_df.index = results_df.index + 1
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.error("❌ Failed to make prediction. Please try again with a different image.")
        
        else:
            st.info("👆 Please upload an image to start classification")
            
            # Show key features
            st.markdown("### 🌟 Key Features")
            
            features = [
                "🎯 YOLO-powered classification",
                "⚡ Real-time inference", 
                "🔧 Adjustable confidence threshold",
                "📊 Visual confidence scores",
                "🎨 Professional interface"
            ]
            
            for feature in features:
                st.markdown(f"- {feature}")
    
    # Footer section
    st.markdown("---")
    st.markdown("### 🚀 Powered by YOLO - Ultralytics ⚡ Created by Sai Sudharsan S G")

if __name__ == "__main__":
    main()