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
        background-color: #0068c9;
        padding: 15px;
        color: white !important;
        border-radius: 10px;
        margin-bottom: 20px;
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
    st.markdown("<h1 class='main-header'>AI-Based Cattle Breed Identification</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sub-header'>
        <p>This application uses a deep learning model to identify cattle breeds from images.</p>
        <p>Upload an image of cattle, and the AI will analyze and predict the breed with confidence scores.</p>
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
            <p class="dark-mode-compatible">This application uses a MobileNetV2-based model trained on Gir and Murrah cattle breeds. 
            Upload an image to get instant breed predictions with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #0068c9; margin-top: 25px;'>Model Information</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <p class="dark-mode-compatible"><strong>🔹 Model:</strong> MobileNetV2 (Transfer Learning)</p>
            <p class="dark-mode-compatible"><strong>🔹 Classes:</strong> Gir, Murrah</p>
            <p class="dark-mode-compatible"><strong>🔹 Input Size:</strong> 224×224 pixels</p>
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
        uploaded_file = st.file_uploader("Upload an image of cattle", type=["jpg", "jpeg", "png"])
        
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
                    else:
                        # Display warning for non-cattle images with detailed metrics
                        st.markdown(f"""
                        <div style="background-color:#FF5151; padding:15px; border-radius:10px; margin-bottom:20px">
                            <h3 style="color:white; margin:0;">⚠️ Not a Recognized Cattle Breed</h3>
                            <p style="color:white; font-size:1.2em; margin:5px 0 0 0;">
                                This image doesn't appear to be one of the cattle breeds this model was trained on.
                            </p>
                            <p style="color:white; font-size:0.9em; margin:10px 0 0 0;">
                                Confidence: {top_conf:.2%}, Prediction ratio: {prediction_result.get('prediction_ratio', 0):.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display all predictions with clearer formatting
                    st.markdown("### All Predictions:")
                    
                    for i, (breed, conf) in enumerate(zip(breeds, confidences)):
                        # Highlight the top prediction differently
                        if i == 0:
                            st.markdown(f"""
                            <div class="prediction-card prediction-primary">
                                <p class="dark-mode-compatible" style="font-size:1.1em; margin:0"><strong>{breed.title()}</strong> - {conf:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-card prediction-secondary">
                                <p class="dark-mode-compatible" style="font-size:1.1em; margin:0"><strong>{breed.title()}</strong> - {conf:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Create and display improved bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    y_pos = range(len(breeds))
                    
                    # Use different colors for the bars based on validation
                    if is_valid_cattle:
                        colors = ['#0068c9', '#fa4b42'][:len(breeds)]
                    else:
                        colors = ['#FF5151', '#FF7F7F'][:len(breeds)]
                    
                    # Create horizontal bar chart with thicker bars
                    bars = ax.barh(y_pos, [conf * 100 for conf in confidences], color=colors, height=0.5)
                    
                    # Add percentage annotations to the bars
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                f'{confidences[i]:.2%}', ha='left', va='center', fontsize=12, fontweight='bold')
                    
                    # Set chart properties with better formatting
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([breed.title() for breed in breeds], fontsize=12)
                    ax.set_xlabel('Confidence (%)', fontsize=12)
                    ax.set_title('Breed Prediction Confidence', fontsize=14, fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Adjust x-axis limit to ensure annotations fit
                    ax.set_xlim(0, 110)
                    
                    # Add threshold line at 85% to indicate confidence threshold
                    ax.axvline(x=85, color='red', linestyle='--', alpha=0.7)
                    ax.text(86, -0.4, 'Threshold (85%)', color='white', fontsize=10, alpha=0.8)
                    
                    # Create a chart that works well in both light and dark modes
                    # Use darker colors that will be visible on both backgrounds
                    
                    # Set a transparent background for the figure 
                    # Use tuple format (R, G, B, A) with values from 0 to 1
                    text_color = '#FFFFFF'  # White text for visibility on dark backgrounds
                    grid_color = '#666666'  # Medium gray grid
                    
                    # Use transparent background (properly formatted for matplotlib)
                    fig.patch.set_facecolor('none')  # Transparent background
                    ax.set_facecolor((0.1, 0.1, 0.1, 0.1))  # Very slight darkening
                    
                    # Use high-contrast colors for text and labels that will be visible in any mode
                    # Keep the original colors of the bars which have good contrast already
                    
                    # Set tick parameters with more visible colors
                    ax.tick_params(axis='both', colors='white')
                    
                    # Make axes labels more visible
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.title.set_color('white')
                    
                    # Ensure all text in the chart is visible
                    for text in ax.texts:
                        text.set_color('white')
                    
                    # Set grid with visible lines
                    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#888888')
                    
                    # Show the chart
                    st.pyplot(fig)
                    
                    # Add "Save Record" button with enhanced styling for dark mode compatibility
                    st.markdown("<hr style='margin: 30px 0; opacity: 0.3;'>", unsafe_allow_html=True)
                    st.markdown("<h3 class='dark-mode-compatible' style='margin-bottom:15px;'>Save Prediction Results</h3>", unsafe_allow_html=True)
                    
                    col_btn1, col_btn2 = st.columns([1, 2])
                    with col_btn1:
                        if st.button("📋 Save Record (BPA Integration)", use_container_width=True, type="primary"):
                            csv_file = save_to_csv(prediction_result, image)
                            st.success(f"✅ Record saved successfully!")
                            
                            # Show latest records
                            if os.path.exists(csv_file):
                                st.markdown("<h4 class='dark-mode-compatible' style='margin:20px 0 10px 0;'>Recent Records</h4>", unsafe_allow_html=True)
                                df = pd.read_csv(csv_file)
                                
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