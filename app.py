"""
Modern Cattle Breed Identification Web App
==========================================
A Streamlit web interface for the 41-breed cattle classifier
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import datetime
import numpy as np
from PIL import Image
import io
import seaborn as sns
import torch
import json
from sklearn.metrics import confusion_matrix, classification_report
from modern_predictor import ModernCattlePredictor

# Set page configuration
st.set_page_config(
    page_title="Cattle Breed Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_data
def load_breed_information():
    """Load breed information from JSON file"""
    try:
        with open('breedInformation.json', 'r', encoding='utf-8') as f:
            breed_data = json.load(f)
        
        # Create name mapping for faster lookup
        breed_data['name_mapping'] = create_breed_name_mapping(breed_data)
        return breed_data
    except FileNotFoundError:
        st.warning("⚠️ Breed information file not found. Breed details will not be available.")
        return {"buffalo_breeds": [], "cattle_breeds": [], "name_mapping": {}}
    except Exception as e:
        st.error(f"❌ Error loading breed information: {str(e)}")
        return {"buffalo_breeds": [], "cattle_breeds": [], "name_mapping": {}}

def normalize_breed_name(name):
    """Normalize breed name for better matching"""
    return name.lower().strip().replace('_', ' ').replace('-', ' ')

def create_breed_name_mapping(breed_data):
    """Create a mapping dictionary for common breed name variations"""
    mapping = {}
    all_breeds = breed_data.get("buffalo_breeds", []) + breed_data.get("cattle_breeds", [])
    
    for breed in all_breeds:
        breed_name = breed["name"]
        
        # Add various formats that models might predict
        variations = [
            breed_name.lower(),
            breed_name.lower().replace(' ', '_'),
            breed_name.lower().replace(' ', ''),
            breed_name.lower().replace('-', '_'),
        ]
        
        for variation in variations:
            mapping[variation] = breed_name
        
        # Add alternative names
        for alt_name in breed.get("also_known_as", []):
            alt_variations = [
                alt_name.lower(),
                alt_name.lower().replace(' ', '_'),
                alt_name.lower().replace(' ', ''),
                alt_name.lower().replace('-', '_'),
            ]
            for alt_var in alt_variations:
                mapping[alt_var] = breed_name
    
    return mapping

def get_breed_info(breed_name, breed_data):
    """Get detailed information for a specific breed with enhanced name matching"""
    breed_name_lower = breed_name.lower().strip()
    
    # First, try the fast mapping lookup
    name_mapping = breed_data.get('name_mapping', {})
    if breed_name_lower in name_mapping:
        actual_breed_name = name_mapping[breed_name_lower]
        # Find the breed with this actual name
        all_breeds = breed_data.get("buffalo_breeds", []) + breed_data.get("cattle_breeds", [])
        for breed in all_breeds:
            if breed["name"] == actual_breed_name:
                return breed
    
    # If mapping didn't work, fall back to detailed matching
    # Create variations of the breed name to handle different formats
    breed_variations = [
        breed_name_lower,
        breed_name_lower.replace('_', ' '),  # krishna_valley -> krishna valley
        breed_name_lower.replace(' ', '_'),  # krishna valley -> krishna_valley
        breed_name_lower.replace('-', ' '),  # handle hyphens
        breed_name_lower.replace(' ', '-'),  # handle hyphens
        breed_name_lower.replace('_', ''),   # remove underscores completely
        breed_name_lower.replace(' ', ''),   # remove spaces completely
    ]
    
    # Search in both buffalo and cattle breeds
    all_breeds = breed_data.get("buffalo_breeds", []) + breed_data.get("cattle_breeds", [])
    
    for breed in all_breeds:
        breed_main_name = breed["name"].lower().strip()
        
        # Create variations for the JSON breed name as well
        json_variations = [
            breed_main_name,
            breed_main_name.replace(' ', '_'),
            breed_main_name.replace('_', ' '),
            breed_main_name.replace('-', ' '),
            breed_main_name.replace(' ', '-'),
            breed_main_name.replace('_', ''),
            breed_main_name.replace(' ', ''),
        ]
        
        # Check if any variation matches
        for pred_var in breed_variations:
            for json_var in json_variations:
                if pred_var == json_var:
                    return breed
        
        # Check alternative names with the same variation logic
        for alt_name in breed.get("also_known_as", []):
            alt_name_lower = alt_name.lower().strip()
            alt_variations = [
                alt_name_lower,
                alt_name_lower.replace(' ', '_'),
                alt_name_lower.replace('_', ' '),
                alt_name_lower.replace('-', ' '),
                alt_name_lower.replace(' ', '-'),
                alt_name_lower.replace('_', ''),
                alt_name_lower.replace(' ', ''),
            ]
            
            for pred_var in breed_variations:
                for alt_var in alt_variations:
                    if pred_var == alt_var:
                        return breed
    
    # If no exact match found, try a more fuzzy approach
    # Normalize both predicted and JSON names to basic form
    normalized_pred = normalize_breed_name(breed_name)
    
    for breed in all_breeds:
        normalized_json = normalize_breed_name(breed["name"])
        
        # Check if normalized names match
        if normalized_pred == normalized_json:
            return breed
        
        # Check alternative names with normalization
        for alt_name in breed.get("also_known_as", []):
            if normalized_pred == normalize_breed_name(alt_name):
                return breed
    
    return None

@st.cache_resource
def load_predictor():
    """Load the cattle breed predictor model - Using Original MobileNetV2!"""
    try:
        # Use Original MobileNetV2 - the TRUE winner with 89.49% validation accuracy!
        model_path = 'mobilenetv2_best_model.pth'
        results_path = 'mobilenetv2_pytorch_results.json'
        
        # Check if required files exist
        if not os.path.exists(model_path):
            st.error("❌ Original MobileNetV2 model file not found. Please train the model first!")
            return None
        if not os.path.exists(results_path):
            st.error("❌ Original MobileNetV2 training results file not found. Please train the model first!")
            return None
            
        predictor = ModernCattlePredictor(model_path, results_path)
        return predictor
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("💡 Make sure you have trained the model by running: python mobilenetv2_pytorch_train.py")
        return None

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location and return the path"""
    # Check if uploaded_file is None
    if uploaded_file is None:
        return None
        
    # Create a temporary directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # Preserve the original filename if available, otherwise generate a unique name
    if hasattr(uploaded_file, 'name'):
        original_filename = uploaded_file.name
    else:
        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"uploaded_image_{timestamp}.jpg"
    
    # Save the uploaded file
    file_path = os.path.join('temp', original_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    print(f"Saved uploaded file: {original_filename} to {file_path}")
    return file_path

def save_to_csv(data, image):
    """Save prediction results to CSV with timestamp and image"""
    # Create records directory if it doesn't exist
    if not os.path.exists('records'):
        os.makedirs('records')
    
    csv_file = os.path.join('records', 'breed_predictions.csv')
    
    # Convert image to bytes for storage
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the data
    new_data = {
        'Timestamp': timestamp,
        'Image_Filename': data.get('filename', 'unknown'),
        'Primary_Breed': data.get('Prediction 1', {}).get('breed', 'unknown'),
        'Primary_Confidence': data.get('Prediction 1', {}).get('confidence', 0),
        'Secondary_Breed': data.get('Prediction 2', {}).get('breed', 'unknown') if 'Prediction 2' in data else '',
        'Secondary_Confidence': data.get('Prediction 2', {}).get('confidence', 0) if 'Prediction 2' in data else 0,
        'Valid_Cattle': data.get('valid_cattle', False)
    }
    
    # Check if file exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    
    return csv_file

def main():
    """Main function to run the Streamlit app"""
    
    # Enhanced CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #0068c9 0%, #004c9c 100%);
        padding: 25px;
        color: white !important;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 1.8em;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3em;
        padding: 15px;
        text-align: center;
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        margin-bottom: 25px;
        color: inherit !important;
    }
    .dark-mode-compatible {
        color: inherit !important;
    }
    .highlight-box {
        background: linear-gradient(135deg, #0068c9 0%, #004c9c 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .warning-box {
        background-color: #FF5151;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white !important;
        box-shadow: 0 4px 8px rgba(255,81,81,0.3);
    }
    .image-border {
        border: 3px solid #0068c9;
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Apply these styles to ensure text is visible in all modes */
    .dark-mode-compatible {
        color: inherit !important;
    }
    .highlight-box {
        background-color: #0068c9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white !important;
    }
    .prediction-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .prediction-primary {
        background-color: rgba(0, 104, 201, 0.15);
        border-left: 5px solid #0068c9;
    }
    .prediction-secondary {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 5px solid #fa4b42;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description with enhanced styling that works in dark mode
    st.markdown("<h1 class='main-header'>🐄 AI-Based Cattle Breed Identification System</h1>", unsafe_allow_html=True)
    
    # Load predictor for sidebar info
    predictor = load_predictor()
    
    # Enhanced sidebar with comprehensive information
    with st.sidebar:
        # Logo/Image placeholder
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <div style='font-size: 4em; margin-bottom: 10px;'>🐄</div>
            <h2 style='color: #0068c9; margin: 0;'>Cattle AI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #0068c9;'>About</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="border-left: 4px solid #0068c9;">
            <p class="dark-mode-compatible">Advanced AI-powered cattle breed identification system using 
            deep learning CNN models. Upload an image to get instant breed predictions with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #0068c9; margin-top: 25px;'>Model Information</h2>", unsafe_allow_html=True)
        if predictor:
            st.markdown(f"""
            <div class="info-box">
                <p class="dark-mode-compatible"><strong>🔹 Model:</strong> Original MobileNetV2</p>
                <p class="dark-mode-compatible"><strong>🔹 Classes:</strong> {len(predictor.classes)} cattle breeds</p>
                <p class="dark-mode-compatible"><strong>🔹 Validation Accuracy:</strong> {predictor.best_accuracy:.1f}%</p>
                <p class="dark-mode-compatible"><strong>🔹 Test Accuracy:</strong> 88.91%</p>
                <p class="dark-mode-compatible"><strong>🔹 Input Size:</strong> 224×224 pixels</p>
                <p class="dark-mode-compatible"><strong>🔹 Device:</strong> {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎉 Model Selection Results!")
            st.markdown("""
            <div class="info-box">
                <p class="dark-mode-compatible">Our comprehensive analysis revealed Original MobileNetV2 
                significantly outperforms enhanced versions:</p>
                <p class="dark-mode-compatible"><strong>✅ Original MobileNetV2:</strong> 89.49%</p>
                <p class="dark-mode-compatible"><strong>❌ Enhanced MobileNetV2:</strong> 59.48%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add help section
        st.markdown("<h2 style='text-align: center; color: #0068c9; margin-top: 25px;'>How to Use</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <ol class="dark-mode-compatible">
                <li>Select the <strong>"Image Classification"</strong> tab</li>
                <li>Click <strong>"Browse files"</strong> button</li>
                <li>Select a cattle image (JPG, PNG)</li>
                <li>View predictions and confidence scores</li>
                <li>Save results to CSV for record keeping</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show supported breeds
        if predictor:
            with st.expander("🐄 Supported Breeds"):
                breeds_df = pd.DataFrame({'Breed': predictor.classes})
                st.dataframe(breeds_df, use_container_width=True)
        else:
            st.error("Model not loaded!")
            
        st.markdown("---")
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Upload clear, well-lit images
        - Ensure the cattle is clearly visible
        - Avoid blurry or distant shots
        - Single animal per image works best
        """)
    st.markdown("""
    <div class='sub-header'>
        <p>This application uses a deep learning CNN model to identify cattle and buffalo breeds from images.</p>
        <p>Upload an image of cattle or buffalo, and the AI will analyze and predict the breed with confidence scores.</p>
        <p><strong>Integration Ready:</strong> Designed for Bharat Pashudhan App (BPA) data standards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for different functions
    tab1, tab2 = st.tabs(["🖼️ Image Classification", "� Model Performance"])
    
    # File upload - inside the Image Classification tab
    with tab1:
        # Check if model is loaded first
        predictor = load_predictor()
        if predictor is None:
            st.warning("⚠️ Model not loaded. Please check the error messages above.")
            st.stop()
            
        uploaded_file = st.file_uploader("Upload an image of cattle or buffalo", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            # No need for inline styles as we've already defined them in the main CSS block
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<h3 style='margin-bottom:15px;'>Uploaded Image</h3>", unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                # Add a border to the image with CSS that works in both light and dark modes
                st.markdown("""
                <div style="border: 2px solid #0068c9; border-radius: 10px; padding: 10px; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                caption = f'File: {uploaded_file.name}' if hasattr(uploaded_file, 'name') else 'Uploaded Image'
                st.image(image, caption=caption, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Save the uploaded file and predict
        with st.spinner('Analyzing image...'):
            # Save the file temporarily
            file_path = save_uploaded_file(uploaded_file)
            
            # Check if file_path is None (which means uploaded_file was None)
            if file_path is None:
                st.error("No file was uploaded or there was an error with the uploaded file.")
                return
                
            try:
                # Print debug info about the file being processed
                if hasattr(uploaded_file, 'name'):
                    st.write(f"Processing file: {uploaded_file.name}")
                else:
                    st.write("Processing uploaded file")
                
                # Load predictor
                predictor = load_predictor()
                if predictor is None:
                    st.error("Failed to load the model!")
                    return
                
                # Make prediction using the new predictor
                result = predictor.predict_single(file_path)
                
                # Convert result to predictions format
                if 'error' in result:
                    st.error(f"❌ Prediction failed: {result['error']}")
                    return
                
                predictions = result['top_predictions']
                
                # Check if predictions were returned
                if not predictions or len(predictions) == 0:
                    st.error("❌ No predictions returned. Please try a different image.")
                    return
                
                # Enhanced confidence threshold validation (85%)
                CONFIDENCE_THRESHOLD = 85.0
                
                # Convert to enhanced format with validation
                prediction_result = {}
                for i, pred in enumerate(predictions[:5], 1):
                    prediction_result[f'Prediction {i}'] = {
                        'breed': pred['breed'],
                        'confidence': pred['percentage'] / 100.0  # Convert to 0-1 scale for compatibility
                    }
                
                # Enhanced validation: use 85% confidence threshold
                top_confidence = predictions[0]['percentage']
                prediction_result['valid_cattle'] = top_confidence >= CONFIDENCE_THRESHOLD
                prediction_result['top_confidence'] = top_confidence
                
                # Extract predictions for display and chart
                breeds = []
                confidences = []
                
                # Add filename to prediction result for CSV
                if uploaded_file and hasattr(uploaded_file, 'name'):
                    prediction_result['filename'] = uploaded_file.name
                
                for key, value in prediction_result.items():
                    if key.startswith('Prediction'):
                        breeds.append(value['breed'])
                        confidences.append(value['confidence'])
                
                with col2:
                    st.markdown("<h3 style='margin-bottom:15px;'>🎯 Prediction Results</h3>", unsafe_allow_html=True)
                    
                    # Check if the image meets confidence threshold
                    is_valid_cattle = prediction_result.get('valid_cattle', False)
                    top_confidence = prediction_result.get('top_confidence', 0)
                    
                    # Enhanced prediction display
                    top_breed = breeds[0] if breeds else "Unknown"
                    top_conf = confidences[0] if confidences else 0
                    
                    # Display results - always show top prediction with confidence status
                    if is_valid_cattle:
                        # High confidence - Display professional results
                        st.markdown(f"""
                        <div class="highlight-box">
                            <h3 style="margin:0;">✅ Predicted Breed: {top_breed.upper()}</h3>
                            <p style="font-size:1.3em; margin:10px 0 0 0;">Confidence: {top_confidence:.1f}%</p>
                            <p style="font-size:1.0em; margin:5px 0 0 0;">Status: High Confidence (≥85%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Low confidence - Still show top prediction but with warning styling
                        st.markdown(f"""
                        <div class="warning-box">
                            <h3 style="color:white; margin:0; text-align:center;">⚠️ Predicted Breed: {top_breed.upper()}</h3>
                            <p style="color:white; font-size:1.2em; margin:15px 0; text-align:center;">
                                Confidence: {top_confidence:.1f}%
                            </p>
                            <p style="color:white; font-size:1.0em; margin:0; text-align:center;">
                                Status: Low Confidence (Below 85% threshold)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Load breed information
                        breed_data = load_breed_information()
                        
                        # Show only top 3 predictions with enhanced design and breed info
                        top_3_breeds = breeds[:3]
                        top_3_confidences = confidences[:3]
                        
                        for i, (breed, conf) in enumerate(zip(top_3_breeds, top_3_confidences)):
                            # Get breed information
                            breed_info = get_breed_info(breed, breed_data)
                            
                            # Debug: Show what we're trying to match (only in development)
                            # st.write(f"Debug: Looking for '{breed}' -> Found: {breed_info is not None}")
                            # Calculate confidence percentage for display
                            conf_percentage = conf * 100 if conf <= 1 else conf
                            
                            # Enhanced styling with ranking and breed information
                            if i == 0:
                                # Gold medal for top prediction
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #0068c9 0%, #004c9c 100%);
                                           padding: 18px; border-radius: 15px; margin: 12px 0;
                                           box-shadow: 0 6px 12px rgba(0,0,0,0.15); border-left: 6px solid #FFD700;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <p style="color: white; font-size: 1.4em; margin: 0; font-weight: bold;">
                                                🥇 #{i+1}: {breed.title()}
                                            </p>
                                            <p style="color: #e8f4f8; font-size: 1.0em; margin: 5px 0 0 0;">
                                                Primary Prediction
                                            </p>
                                        </div>
                                        <div style="text-align: right;">
                                            <p style="color: #FFD700; font-size: 1.8em; margin: 0; font-weight: bold;">
                                                {conf_percentage:.1f}%
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add detailed breed information for top prediction
                                with st.expander(f"📖 Learn more about {breed.title()}", expanded=True):
                                    if breed_info:
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.markdown("**🏞️ Origin & Location:**")
                                            if breed_info.get("breeding_tract"):
                                                tract = breed_info["breeding_tract"]
                                                st.write(f"• **State:** {tract.get('state', 'Not specified')}")
                                                if tract.get('districts'):
                                                    districts = ", ".join(tract["districts"][:3])  # Show first 3 districts
                                                    if len(tract["districts"]) > 3:
                                                        districts += f" and {len(tract['districts']) - 3} more"
                                                    st.write(f"• **Districts:** {districts}")
                                            
                                            if breed_info.get("also_known_as"):
                                                st.markdown("**📝 Also Known As:**")
                                                st.write(f"• {', '.join(breed_info['also_known_as'])}")
                                        
                                        with col2:
                                            st.markdown("**🐄 Physical Characteristics:**")
                                            if breed_info.get("physical_characteristics"):
                                                phys = breed_info["physical_characteristics"]
                                                if phys.get("color"):
                                                    st.write(f"• **Color:** {phys['color']}")
                                                if phys.get("horns"):
                                                    st.write(f"• **Horns:** {phys['horns']}")
                                                if phys.get("size_and_body"):
                                                    st.write(f"• **Body:** {phys['size_and_body']}")
                                        
                                        # Additional information in full width
                                        if breed_info.get("milk_yield"):
                                            st.markdown("**🥛 Milk Production:**")
                                            st.info(breed_info["milk_yield"])
                                        
                                        if breed_info.get("utility"):
                                            st.markdown("**🎯 Primary Use:**")
                                            st.success(breed_info["utility"])
                                        
                                        if breed_info.get("special_facts"):
                                            st.markdown("**✨ Special Facts:**")
                                            st.warning(breed_info["special_facts"])
                                    else:
                                        st.info(f"📚 Detailed information for **{breed.title()}** is not available in our database yet. We are continuously updating our breed information.")
                            elif i == 1:
                                # Silver medal for second prediction
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                                           padding: 15px; border-radius: 12px; margin: 10px 0;
                                           box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 5px solid #C0C0C0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <p style="color: white; font-size: 1.2em; margin: 0; font-weight: bold;">
                                                � #{i+1}: {breed.title()}
                                            </p>
                                            <p style="color: #e0e0e0; font-size: 0.9em; margin: 3px 0 0 0;">
                                                Alternative Prediction
                                            </p>
                                        </div>
                                        <div style="text-align: right;">
                                            <p style="color: #C0C0C0; font-size: 1.5em; margin: 0; font-weight: bold;">
                                                {conf_percentage:.1f}%
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add compact breed information for second prediction
                                with st.expander(f"📋 Quick info about {breed.title()}", expanded=False):
                                    if breed_info:
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            if breed_info.get("breeding_tract", {}).get("state"):
                                                st.write(f"**📍 Found in:** {breed_info['breeding_tract']['state']}")
                                            if breed_info.get("physical_characteristics", {}).get("color"):
                                                color_info = breed_info['physical_characteristics']['color']
                                                color_short = color_info[:100] + "..." if len(color_info) > 100 else color_info
                                                st.write(f"**🎨 Color:** {color_short}")
                                        
                                        with col2:
                                            if breed_info.get("milk_yield"):
                                                milk_info = breed_info["milk_yield"]
                                                milk_short = milk_info[:80] + "..." if len(milk_info) > 80 else milk_info
                                                st.write(f"**🥛 Milk:** {milk_short}")
                                            if breed_info.get("utility"):
                                                utility_info = breed_info["utility"]
                                                utility_short = utility_info[:60] + "..." if len(utility_info) > 60 else utility_info
                                                st.write(f"**🎯 Use:** {utility_short}")
                                    else:
                                        st.info(f"📚 Information for **{breed.title()}** coming soon!")
                            else:
                                # Bronze medal for third prediction
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
                                           padding: 12px; border-radius: 10px; margin: 8px 0;
                                           box-shadow: 0 3px 6px rgba(0,0,0,0.1); border-left: 4px solid #CD7F32;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <p style="color: white; font-size: 1.1em; margin: 0; font-weight: 600;">
                                                🥉 #{i+1}: {breed.title()}
                                            </p>
                                            <p style="color: #f0e6d2; font-size: 0.85em; margin: 2px 0 0 0;">
                                                Third Option
                                            </p>
                                        </div>
                                        <div style="text-align: right;">
                                            <p style="color: #CD7F32; font-size: 1.3em; margin: 0; font-weight: bold;">
                                                {conf_percentage:.1f}%
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add brief breed information for third prediction
                                with st.expander(f"ℹ️ Brief info about {breed.title()}", expanded=False):
                                    if breed_info:
                                        if breed_info.get("breeding_tract", {}).get("state"):
                                            st.write(f"**📍 Origin:** {breed_info['breeding_tract']['state']}")
                                        if breed_info.get("utility"):
                                            st.write(f"**🎯 Primary Use:** {breed_info['utility']}")
                                        if breed_info.get("special_facts"):
                                            facts = breed_info["special_facts"]
                                            facts_short = facts[:120] + "..." if len(facts) > 120 else facts
                                            st.write(f"**✨ Special:** {facts_short}")
                                    else:
                                        st.info(f"📚 Info for **{breed.title()}** not available yet.")
                    
                    # UNIVERSAL TOP 3 PREDICTIONS DISPLAY - Always shown regardless of confidence
                    # Load breed information
                    breed_data = load_breed_information()
                    
                    # Show top 3 predictions with enhanced design and breed info
                    top_3_breeds = breeds[:3]
                    top_3_confidences = confidences[:3]
                    
                    for i, (breed, conf) in enumerate(zip(top_3_breeds, top_3_confidences)):
                        # Get breed information
                        breed_info = get_breed_info(breed, breed_data)
                        
                        # Calculate confidence percentage for display
                        conf_percentage = conf * 100 if conf <= 1 else conf
                        
                        # Enhanced styling with ranking and breed information
                        if i == 0:
                            # Gold medal for top prediction
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #0068c9 0%, #004c9c 100%);
                                       padding: 18px; border-radius: 15px; margin: 12px 0;
                                       box-shadow: 0 6px 12px rgba(0,0,0,0.15); border-left: 6px solid #FFD700;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <p style="color: white; font-size: 1.4em; margin: 0; font-weight: bold;">
                                            🥇 #{i+1}: {breed.title()}
                                        </p>
                                        <p style="color: #e8f4f8; font-size: 1.0em; margin: 5px 0 0 0;">
                                            Primary Prediction
                                        </p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="color: #FFD700; font-size: 1.8em; margin: 0; font-weight: bold;">
                                            {conf_percentage:.1f}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add detailed breed information for top prediction (expanded for high confidence)
                            expanded_state = is_valid_cattle  # Expand for high confidence, collapse for low confidence
                            with st.expander(f"📖 Learn more about {breed.title()}", expanded=expanded_state):
                                if breed_info:
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.markdown("**🏞️ Origin & Location:**")
                                        if breed_info.get("breeding_tract"):
                                            tract = breed_info["breeding_tract"]
                                            st.write(f"• **State:** {tract.get('state', 'Not specified')}")
                                            if tract.get('districts'):
                                                districts = ", ".join(tract["districts"][:3])  # Show first 3 districts
                                                if len(tract["districts"]) > 3:
                                                    districts += f" and {len(tract['districts']) - 3} more"
                                                st.write(f"• **Districts:** {districts}")
                                        
                                        if breed_info.get("also_known_as"):
                                            st.markdown("**📝 Also Known As:**")
                                            st.write(f"• {', '.join(breed_info['also_known_as'])}")
                                    
                                    with col2:
                                        st.markdown("**🐄 Physical Characteristics:**")
                                        if breed_info.get("physical_characteristics"):
                                            phys = breed_info["physical_characteristics"]
                                            if phys.get("color"):
                                                st.write(f"• **Color:** {phys['color']}")
                                            if phys.get("horns"):
                                                st.write(f"• **Horns:** {phys['horns']}")
                                            if phys.get("size_and_body"):
                                                st.write(f"• **Body:** {phys['size_and_body']}")
                                    
                                    # Additional information in full width
                                    if breed_info.get("milk_yield"):
                                        st.markdown("**🥛 Milk Production:**")
                                        st.info(breed_info["milk_yield"])
                                    
                                    if breed_info.get("utility"):
                                        st.markdown("**🎯 Primary Use:**")
                                        st.success(breed_info["utility"])
                                    
                                    if breed_info.get("special_facts"):
                                        st.markdown("**✨ Special Facts:**")
                                        st.warning(breed_info["special_facts"])
                                else:
                                    st.info(f"📚 Detailed information for **{breed.title()}** is not available in our database yet. We are continuously updating our breed information.")
                        elif i == 1:
                            # Silver medal for second prediction
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                                       padding: 15px; border-radius: 12px; margin: 10px 0;
                                       box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 5px solid #C0C0C0;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <p style="color: white; font-size: 1.2em; margin: 0; font-weight: bold;">
                                            🥈 #{i+1}: {breed.title()}
                                        </p>
                                        <p style="color: #e0e0e0; font-size: 0.9em; margin: 3px 0 0 0;">
                                            Alternative Prediction
                                        </p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="color: #C0C0C0; font-size: 1.5em; margin: 0; font-weight: bold;">
                                            {conf_percentage:.1f}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add compact breed information for second prediction
                            with st.expander(f"📋 Quick info about {breed.title()}", expanded=False):
                                if breed_info:
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        if breed_info.get("breeding_tract", {}).get("state"):
                                            st.write(f"**📍 Found in:** {breed_info['breeding_tract']['state']}")
                                        if breed_info.get("physical_characteristics", {}).get("color"):
                                            color_info = breed_info['physical_characteristics']['color']
                                            color_short = color_info[:100] + "..." if len(color_info) > 100 else color_info
                                            st.write(f"**🎨 Color:** {color_short}")
                                    
                                    with col2:
                                        if breed_info.get("milk_yield"):
                                            milk_info = breed_info["milk_yield"]
                                            milk_short = milk_info[:80] + "..." if len(milk_info) > 80 else milk_info
                                            st.write(f"**🥛 Milk:** {milk_short}")
                                        if breed_info.get("utility"):
                                            utility_info = breed_info["utility"]
                                            utility_short = utility_info[:60] + "..." if len(utility_info) > 60 else utility_info
                                            st.write(f"**🎯 Use:** {utility_short}")
                                else:
                                    st.info(f"📚 Information for **{breed.title()}** coming soon!")
                        else:
                            # Bronze medal for third prediction
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
                                       padding: 12px; border-radius: 10px; margin: 8px 0;
                                       box-shadow: 0 3px 6px rgba(0,0,0,0.1); border-left: 4px solid #CD7F32;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <p style="color: white; font-size: 1.1em; margin: 0; font-weight: 600;">
                                            🥉 #{i+1}: {breed.title()}
                                        </p>
                                        <p style="color: #f0e6d2; font-size: 0.85em; margin: 2px 0 0 0;">
                                            Third Option
                                        </p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="color: #CD7F32; font-size: 1.3em; margin: 0; font-weight: bold;">
                                            {conf_percentage:.1f}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add brief breed information for third prediction
                            with st.expander(f"ℹ️ Brief info about {breed.title()}", expanded=False):
                                if breed_info:
                                    if breed_info.get("breeding_tract", {}).get("state"):
                                        st.write(f"**📍 Origin:** {breed_info['breeding_tract']['state']}")
                                    if breed_info.get("utility"):
                                        st.write(f"**🎯 Primary Use:** {breed_info['utility']}")
                                    if breed_info.get("special_facts"):
                                        facts = breed_info["special_facts"]
                                        facts_short = facts[:120] + "..." if len(facts) > 120 else facts
                                        st.write(f"**✨ Special:** {facts_short}")
                                else:
                                    st.info(f"📚 Info for **{breed.title()}** not available yet.")
                    
                    # Create enhanced confidence chart for top 3 predictions
                    if breeds and confidences:
                        st.markdown("<h4 style='margin: 25px 0 15px 0;'>📊 Top 3 Confidence Visualization:</h4>", unsafe_allow_html=True)
                        
                        # Focus on top 3 predictions only
                        top_3_breeds = breeds[:3]
                        top_3_confidences = confidences[:3]
                        
                        # Create horizontal bar chart with enhanced styling
                        fig, ax = plt.subplots(figsize=(12, 5))
                        y_pos = range(len(top_3_breeds))
                        
                        # Enhanced color scheme for top 3 with ranking theme
                        ranking_colors = ['#0068c9', '#6c757d', '#8B4513']  # Gold, Silver, Bronze theme
                        
                        # Convert confidences to percentages for display
                        conf_percentages = [c * 100 if c <= 1 else c for c in top_3_confidences]
                        
                        # Create bars with ranking colors
                        bars = ax.barh(y_pos, conf_percentages, color=ranking_colors, height=0.7, alpha=0.9)
                        
                        # Add enhanced percentage annotations with ranking medals
                        medals = ['🥇', '🥈', '🥉']
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            # Place text inside bar if wide enough, otherwise outside
                            if width > 30:
                                ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                                        f'{medals[i]} {conf_percentages[i]:.1f}%', 
                                        ha='center', va='center', 
                                        fontsize=14, fontweight='bold', color='white')
                            else:
                                ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                                        f'{medals[i]} {conf_percentages[i]:.1f}%', 
                                        ha='left', va='center', 
                                        fontsize=14, fontweight='bold', color='black')
                        
                        # Enhanced chart properties
                        ax.set_yticks(y_pos)
                        breed_labels = [f"#{i+1}: {breed.title()}" for i, breed in enumerate(top_3_breeds)]
                        ax.set_yticklabels(breed_labels, fontsize=13, fontweight='bold')
                        ax.set_xlabel('Confidence (%)', fontsize=14, fontweight='bold')
                        ax.set_title('🏆 Top 3 Breed Predictions - Confidence Levels', fontsize=16, fontweight='bold', pad=25)
                        
                        # Add 85% threshold line with enhanced styling
                        ax.axvline(x=85, color='#e74c3c', linestyle='--', linewidth=3, alpha=0.9)
                        ax.text(87, 1, 'Confidence\nThreshold\n(85%)', 
                                color='#e74c3c', fontsize=11, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                         edgecolor='#e74c3c', alpha=0.9))
                        
                        # Enhanced styling
                        ax.set_xlim(0, 105)
                        ax.set_facecolor('#f8f9fa')
                        ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.8)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(2)
                        ax.spines['bottom'].set_linewidth(2)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        
                        # Add a quick summary table for top 3 predictions
                        st.markdown("<h4 style='margin: 20px 0 10px 0;'>📋 Prediction Summary:</h4>", unsafe_allow_html=True)
                        
                        # Create summary DataFrame
                        summary_data = []
                        for i, (breed, conf) in enumerate(zip(top_3_breeds, top_3_confidences)):
                            conf_percentage = conf * 100 if conf <= 1 else conf
                            medal = ['🥇 1st', '🥈 2nd', '🥉 3rd'][i]
                            status = "✅ High" if conf_percentage >= 85 else "⚠️ Low"
                            
                            summary_data.append({
                                'Rank': medal,
                                'Breed': breed.title(),
                                'Confidence': f"{conf_percentage:.1f}%",
                                'Status': status
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Display with custom styling
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Rank": st.column_config.TextColumn(
                                    "Rank",
                                    width="small",
                                ),
                                "Breed": st.column_config.TextColumn(
                                    "Predicted Breed",
                                    width="medium",
                                ),
                                "Confidence": st.column_config.TextColumn(
                                    "Confidence Score",
                                    width="small",
                                ),
                                "Status": st.column_config.TextColumn(
                                    "Confidence Level",
                                    width="small",
                                )
                            }
                        )
                    
                    # Save results section (only for high confidence predictions)
                    if is_valid_cattle:
                        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
                        
                        # Professional save section
                        st.markdown("""
                        <div class="highlight-box">
                            <h3 style="color: white; margin: 0; text-align: center;">
                                💾 Save Classification Results
                            </h3>
                            <p style="color: #e8f4f8; margin: 8px 0 0 0; text-align: center;">
                                Save prediction results to CSV database
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add save button
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("💾 Save Classification Results", 
                                        key="save_results",
                                        help="Save prediction results to CSV database",
                                        use_container_width=True, 
                                        type="primary"):
                                csv_file = save_to_csv(prediction_result, image)
                                st.balloons()
                                st.success("✅ Results saved successfully!")
                                
                                # Show recent records
                                if os.path.exists(csv_file):
                                    st.markdown("<h4 style='margin:20px 0 10px 0;'>📋 Recent Records</h4>", unsafe_allow_html=True)
                                    df = pd.read_csv(csv_file)
                                    
                                    # Display last 3 records
                                    display_df = df.tail(3)[['Timestamp', 'Image_Filename', 'Primary_Breed', 'Primary_Confidence']]
                                    display_df['Primary_Confidence'] = display_df['Primary_Confidence'].apply(lambda x: f"{x:.1%}")
                                    
                                    st.dataframe(
                                        display_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )

            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)

    # Model Performance Tab
    with tab2:
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>📊 Model Performance Analysis</h2>", unsafe_allow_html=True)
        
        # Model information
        predictor = load_predictor()
        if predictor:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>🎯 Model Accuracy</h4>
                    <p><strong>Validation:</strong> {:.1f}%</p>
                    <p><strong>Test:</strong> 88.91%</p>
                </div>
                """.format(predictor.best_accuracy), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <h4>📝 Model Details</h4>
                    <p><strong>Architecture:</strong> MobileNetV2</p>
                    <p><strong>Classes:</strong> {len(predictor.classes)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="info-box">
                    <h4>⚙️ Configuration</h4>
                    <p><strong>Input Size:</strong> 224×224</p>
                    <p><strong>Threshold:</strong> 85%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            st.markdown("<h3 style='margin-top: 30px;'>📋 Supported Cattle Breeds</h3>", unsafe_allow_html=True)
            
            # Create a DataFrame with breed information
            breeds_info = []
            for i, breed in enumerate(predictor.classes):
                breeds_info.append({
                    'Breed Name': breed.title(),
                    'Index': i,
                    'Category': 'Cattle Breed'
                })
            
            breeds_df = pd.DataFrame(breeds_info)
            st.dataframe(breeds_df, use_container_width=True, hide_index=True)
            
            # Performance comparison
            st.markdown("<h3 style='margin-top: 30px;'>📈 Model Comparison Results</h3>", unsafe_allow_html=True)
            
            comparison_data = {
                'Model Version': ['Original MobileNetV2', 'Enhanced MobileNetV2'],
                'Validation Accuracy (%)': [89.49, 59.48],
                'Status': ['✅ Selected', '❌ Rejected'],
                'Performance': ['High', 'Low']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models = comparison_data['Model Version']
            accuracies = comparison_data['Validation Accuracy (%)']
            colors = ['#0068c9', '#fa4b42']
            
            bars = ax.bar(models, accuracies, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
            ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # Add threshold line
            ax.axhline(y=85, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(0.5, 87, 'Confidence Threshold (85%)', ha='center', 
                    color='#e74c3c', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        else:
            st.error("Model not loaded. Please check the model files.")
        
        st.subheader("Confusion Matrix")
        st.write("""
        The confusion matrix shows how well our model performs on the test dataset.
        It compares the true labels (actual breeds) with the predicted labels.
        """)
        
        # Add button to generate confusion matrix
        if st.button("Generate Confusion Matrix"):
            with st.spinner("Generating confusion matrix..."):
                try:
                    # Initialize lists to store true and predicted labels
                    y_true = []
                    y_pred = []
                    
                    # Test directory
                    test_dir = 'data/test'
                    
                    # Check if test directory exists
                    if not os.path.exists(test_dir):
                        st.error(f"Test directory '{test_dir}' not found.")
                    else:
                        # Get all image files from test directory
                        image_files = [f for f in os.listdir(test_dir) 
                                    if os.path.isfile(os.path.join(test_dir, f)) and 
                                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        if not image_files:
                            st.warning("No test images found.")
                        else:
                            # Process each test image
                            test_results = []
                            
                            for img_file in image_files:
                                img_path = os.path.join(test_dir, img_file)
                                
                                # Determine true label from filename
                                true_label = None
                                if "gir" in img_file.lower():
                                    true_label = "gir"
                                elif "mur" in img_file.lower():
                                    true_label = "murrah"
                                else:
                                    # For images like dog.jpg that aren't cattle
                                    true_label = "non_cattle"
                                
                                # Get prediction
                                try:
                                    predictor = load_predictor()
                                    result = predictor.predict_single(img_path)
                                    
                                    if 'error' in result:
                                        predictions = []
                                    else:
                                        predictions = result['top_predictions']
                                    
                                    # Get predicted breed
                                    if predictions:
                                        pred_breed = predictions[0]['breed']
                                        confidence = predictions[0]['percentage']
                                    else:
                                        pred_breed = "unknown"
                                        confidence = 0.0
                                    
                                    # Check if it's valid cattle (confidence > 50%)
                                    is_valid_cattle = confidence > 50
                                    
                                    # If not valid cattle, mark as non_cattle
                                    if not is_valid_cattle:
                                        pred_breed = "non_cattle"
                                    
                                    # Store results
                                    y_true.append(true_label)
                                    y_pred.append(pred_breed)
                                    
                                    # Add to test results for display
                                    test_results.append({
                                        'Filename': img_file,
                                        'True Label': true_label,
                                        'Predicted': pred_breed,
                                        'Confidence': confidence,
                                        'Is Valid Cattle': is_valid_cattle
                                    })
                                except Exception as e:
                                    st.error(f"Error processing {img_file}: {e}")
                            
                            # Display test results table
                            st.subheader("Test Results")
                            results_df = pd.DataFrame(test_results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Generate confusion matrix
                            classes = ["gir", "murrah", "non_cattle"]
                            
                            # Ensure all classes are in y_true and y_pred
                            all_classes_y_true = [label if label in classes else "non_cattle" for label in y_true]
                            all_classes_y_pred = [label if label in classes else "non_cattle" for label in y_pred]
                            
                            # Create confusion matrix
                            cm = confusion_matrix(all_classes_y_true, all_classes_y_pred, labels=classes)
                            
                            # Plot confusion matrix
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                        xticklabels=classes, yticklabels=classes)
                            plt.xlabel('Predicted')
                            plt.ylabel('True')
                            plt.title('Confusion Matrix for Cattle Breed Classifier')
                            
                            # Show the confusion matrix
                            st.pyplot(fig)
                            
                            # Display classification report
                            st.subheader("Classification Report")
                            report = classification_report(all_classes_y_true, all_classes_y_pred, 
                                                        labels=classes, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating confusion matrix: {str(e)}")

if __name__ == "__main__":
    main()