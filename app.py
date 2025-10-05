"""
Streamlit App for A    # Save the uploaded file to a temporary location and return the path
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
    return file_pathBreed Identification
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import datetime
import numpy as np
from PIL import Image
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from predict_breed import predict_breed

# Set page configuration
st.set_page_config(
    page_title="AI-Based Cattle Breed Identification",
    layout="wide"
)

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
    
    # Custom CSS for styling that works in both light and dark modes
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
        padding: 10px;
        text-align: center;
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 5px;
        margin-bottom: 25px;
        color: inherit !important;
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
    st.markdown("<h1 class='main-header'>AI-Based Breed Identification System for Cattle and Buffaloes</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sub-header'>
        <p>This application uses a deep learning CNN model to identify cattle and buffalo breeds from images.</p>
        <p>Upload an image of cattle or buffalo, and the AI will analyze and predict the breed with confidence scores.</p>
        <p><strong>Integration Ready:</strong> Designed for Bharat Pashudhan App (BPA) data standards</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for different functions
    tab1, tab2 = st.tabs(["Image Classification", "Model Performance"])
    
    # Enhanced sidebar with information
    with st.sidebar:
        # Logo/Image placeholder
        st.image("https://img.icons8.com/color/96/cow.png", width=100)
        
        st.markdown("<h2 style='text-align: center; color: #0068c9;'>About</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="border-left: 4px solid #0068c9;">
            <p class="dark-mode-compatible">This application uses a MobileNetV2-based CNN model trained on cattle and buffalo breeds (Gir cattle and Murrah buffalo). 
            Upload an image to get instant breed predictions with confidence scores. Designed for BPA integration.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #0068c9; margin-top: 25px;'>Model Information</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p class="dark-mode-compatible"><strong>🔹 Model:</strong> CNN-based (MobileNetV2 Transfer Learning)</p>
            <p class="dark-mode-compatible"><strong>🔹 Classes:</strong> Gir (Cattle), Murrah (Buffalo)</p>
            <p class="dark-mode-compatible"><strong>🔹 Input Size:</strong> 224×224 pixels</p>
            <p class="dark-mode-compatible"><strong>🔹 BPA Ready:</strong> Standardized data format</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a help section
        st.markdown("<h2 style='text-align: center; color: #0068c9; margin-top: 25px;'>How to Use</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <ol class="dark-mode-compatible">
                <li>Click the <strong>"Browse files"</strong> button</li>
                <li>Select a cattle image (JPG, PNG)</li>
                <li>View the predictions and confidence scores</li>
                <li>Save the results using the button below</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload - inside the Image Classification tab
    with tab1:
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
                
                # Make prediction with validation using confidence threshold
                prediction_result = predict_breed(file_path, confidence_threshold=0.85)
                
                # Debug info - show validation result
                st.write(f"Valid cattle: {prediction_result.get('valid_cattle', False)}")
                
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
                    st.subheader("Prediction Results")
                    
                    # Check if the image is likely a cattle breed
                    is_valid_cattle = prediction_result.get('valid_cattle', False)
                    
                    # Create a highlighted box for the top prediction
                    top_breed = breeds[0] if breeds else "Unknown"
                    top_conf = confidences[0] if confidences else 0
                    
                    # Display different messages based on validation
                    if is_valid_cattle:
                        # Display the predicted breed prominently
                        st.markdown(f"""
                        <div class="highlight-box">
                            <h3 style="margin:0;">Predicted Breed: {top_breed.upper()}</h3>
                            <p style="font-size:1.2em; margin:5px 0 0 0;">Confidence: {top_conf:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display all predictions with professional styling
                        st.markdown("<h3 style='margin: 25px 0 15px 0; color: #2c3e50;'>🎯 Classification Results:</h3>", unsafe_allow_html=True)
                        
                        for i, (breed, conf) in enumerate(zip(breeds, confidences)):
                            # Determine breed type
                            breed_type = "(Cattle)" if breed.lower() == "gir" else "(Buffalo)"
                            
                            # Highlight the top prediction differently with better styling
                            if i == 0:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #0068c9 0%, #004c9c 100%);
                                           padding: 15px; border-radius: 12px; margin: 10px 0;
                                           box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FFD700;">
                                    <p style="color: white; font-size: 1.3em; margin: 0; font-weight: bold;">
                                        {breed.title()} {breed_type} - {conf:.1%}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                                           padding: 12px; border-radius: 10px; margin: 8px 0;
                                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                    <p style="color: white; font-size: 1.1em; margin: 0; font-weight: 600;">
                                        {breed.title()} {breed_type} - {conf:.1%}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Create and display improved bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        y_pos = range(len(breeds))
                        
                        # Use different colors for the bars based on validation
                        colors = ['#0068c9', '#fa4b42'][:len(breeds)]
                    else:
                        # Display warning for non-cattle images with detailed metrics
                        st.markdown(f"""
                        <div style="background-color:#FF5151; padding:25px; border-radius:10px; margin-bottom:20px">
                            <h2 style="color:white; margin:0; text-align:center;">⚠️ Not a Recognized Cattle Breed</h2>
                            <p style="color:white; font-size:1.4em; margin:15px 0 0 0; text-align:center;">
                                This image doesn't appear to be one of the cattle breeds this model was trained on.
                            </p>
                            <hr style="border-color:rgba(255,255,255,0.3); margin:20px 0;">
                            <p style="color:white; font-size:1.0em; margin:10px 0 0 0;">
                                This model is specifically trained to identify Gir cattle and Murrah buffalo breeds.
                                For best results, please upload clear images of these livestock breeds.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add sample images section for non-cattle images
                        st.markdown("""
                        <div style="margin-top:30px; padding:15px; background-color:rgba(128, 128, 128, 0.1); border-radius:10px;">
                            <h3 style="text-align:center;">Sample Images</h3>
                            <p style="text-align:center;">Here are examples of cattle breeds this model can identify:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display sample images
                        sample_col1, sample_col2 = st.columns(2)
                        with sample_col1:
                            st.markdown("<h4 style='text-align:center;'>Gir (Cattle)</h4>", unsafe_allow_html=True)
                            if os.path.exists("data/train/gir/Gir1.jpg"):
                                st.image("data/train/gir/Gir1.jpg", caption="Sample Gir cattle breed")
                        with sample_col2:
                            st.markdown("<h4 style='text-align:center;'>Murrah (Buffalo)</h4>", unsafe_allow_html=True)
                            if os.path.exists("data/train/murrah/mur1.jpg"):
                                st.image("data/train/murrah/mur1.jpg", caption="Sample Murrah buffalo breed")
                        
                        # Add a button to try again
                        st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
                        if st.button("Try Again with a Different Image", use_container_width=True):
                            st.experimental_rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Don't show predictions or charts for non-cattle images
                    
                    # Only create and display chart for valid cattle images
                    if is_valid_cattle:
                        # Create horizontal bar chart with better styling
                        bars = ax.barh(y_pos, [conf * 100 for conf in confidences], color=colors, height=0.6, alpha=0.8)
                        
                        # Add percentage annotations to the bars with better positioning
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            # Place text inside the bar if it's wide enough, otherwise outside
                            if width > 30:
                                ax.text(width/2, bar.get_y() + bar.get_height()/2, 
                                        f'{confidences[i]:.1%}', ha='center', va='center', 
                                        fontsize=14, fontweight='bold', color='white')
                            else:
                                ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                                        f'{confidences[i]:.1%}', ha='left', va='center', 
                                        fontsize=14, fontweight='bold', color='black')
                        
                        # Set chart properties with better formatting
                        ax.set_yticks(y_pos)
                        breed_labels = []
                        for breed in breeds:
                            if breed.lower() == 'gir':
                                breed_labels.append('Gir (Cattle)')
                            elif breed.lower() == 'murrah':
                                breed_labels.append('Murrah (Buffalo)')
                            else:
                                breed_labels.append(breed.title())
                        ax.set_yticklabels(breed_labels, fontsize=14, fontweight='bold')
                        ax.set_xlabel('Confidence (%)', fontsize=14, fontweight='bold')
                        ax.set_title('🎯 Breed Classification Confidence', fontsize=16, fontweight='bold', pad=20)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_linewidth(2)
                        ax.spines['bottom'].set_linewidth(2)
                        ax.grid(axis='x', linestyle='-', alpha=0.3, linewidth=1)
                        
                        # Adjust x-axis limit and add better ticks
                        ax.set_xlim(0, 105)
                        ax.set_xticks(range(0, 101, 20))
                        
                        # Add a subtle background
                        ax.set_facecolor('#f8f9fa')
                    
                        # Add threshold line at 85% to indicate confidence threshold
                        ax.axvline(x=85, color='#e74c3c', linestyle='--', linewidth=3, alpha=0.9)
                        
                        # Position threshold text better
                        if len(breeds) > 1:
                            text_y = 0.5
                        else:
                            text_y = 0
                        
                        ax.text(87, text_y, '← Confidence\nThreshold\n(85%)', 
                                color='#e74c3c', fontsize=10, fontweight='bold', 
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                         edgecolor='#e74c3c', alpha=0.9, linewidth=2))
                        
                        # Set professional styling
                        fig.patch.set_facecolor('white')
                        
                        # Set tick parameters with better colors for readability
                        ax.tick_params(axis='both', colors='#2c3e50', labelsize=12)
                        
                        # Make axes labels more visible with professional colors
                        ax.xaxis.label.set_color('#2c3e50')
                        ax.yaxis.label.set_color('#2c3e50')
                        ax.title.set_color('#2c3e50')
                        
                        # Add some padding and adjust layout
                        plt.tight_layout()
                        
                        # Show the chart with better formatting
                        st.pyplot(fig, use_container_width=True)
                    
                    # Add "Save Record" button only for valid cattle images
                    if is_valid_cattle:
                        st.markdown("<hr style='margin: 40px 0; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
                        
                        # Create a professional save section
                        st.markdown("""
                        <div style="background: linear-gradient(90deg, #0068c9 0%, #004c9c 100%); 
                                    padding: 20px; border-radius: 15px; margin: 20px 0;">
                            <h3 style="color: white; margin: 0; text-align: center; font-size: 1.3em;">
                                💾 Save to Bharat Pashudhan App (BPA)
                            </h3>
                            <p style="color: #e8f4f8; margin: 8px 0 0 0; text-align: center; font-size: 1.0em;">
                                Save classification results in BPA-compatible format
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Center the button
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("📋 Save to BPA Database", 
                                        key="save_bpa",
                                        help="Save prediction results to BPA-compatible CSV format",
                                        use_container_width=True, 
                                        type="primary"):
                                csv_file = save_to_csv(prediction_result, image)
                                st.balloons()
                                st.success("✅ Successfully saved to BPA-compatible format!")
                            
                                # Show latest records
                                if os.path.exists(csv_file):
                                    st.markdown("<h4 class='dark-mode-compatible' style='margin:20px 0 10px 0;'>Recent Records</h4>", unsafe_allow_html=True)
                                    df = pd.read_csv(csv_file)
                                    
                                    # Filter to only show valid cattle records
                                    if 'Valid_Cattle' in df.columns:
                                        df = df[df['Valid_Cattle'] == True]
                                    
                                    # Format the dataframe for better display
                                    display_df = df.tail(5)[['Timestamp', 'Image_Filename', 'Primary_Breed', 'Primary_Confidence']]
                                    display_df = display_df.rename(columns={
                                        'Timestamp': 'Date & Time',
                                        'Image_Filename': 'Image',
                                        'Primary_Breed': 'Breed',
                                        'Primary_Confidence': 'Confidence'
                                    })
                                    
                                    # Format confidence as percentage
                                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                                    
                                    # Display styled dataframe
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
        st.header("Model Performance Analysis")
        
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
                                    result = predict_breed(img_path)
                                    
                                    # Get predicted breed
                                    if "Prediction 1" in result:
                                        pred_breed = result["Prediction 1"]["breed"]
                                        confidence = result["Prediction 1"]["confidence"]
                                    else:
                                        pred_breed = "unknown"
                                        confidence = 0.0
                                    
                                    # Check if it's valid cattle
                                    is_valid_cattle = result.get('valid_cattle', True)
                                    
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