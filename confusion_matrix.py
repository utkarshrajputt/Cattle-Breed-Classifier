"""
Generate a confusion matrix for the cattle breed classifier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from predict_breed import predict_breed

def generate_confusion_matrix():
    """Generate and display confusion matrix for test images"""
    
    print("Generating confusion matrix for the cattle breed classifier...")
    
    # Test directory
    test_dir = 'data/test'
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found.")
        return
    
    # Get all image files from test directory
    image_files = [f for f in os.listdir(test_dir) 
                  if os.path.isfile(os.path.join(test_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No test images found.")
        return
    
    # Lists to store true labels and predictions
    y_true = []
    y_pred = []
    confidences = []
    is_valid_cattle_list = []
    filenames = []
    
    # Process each test image
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
            confidences.append(confidence)
            is_valid_cattle_list.append(is_valid_cattle)
            filenames.append(img_file)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Generate detailed results table
    results_df = pd.DataFrame({
        'Filename': filenames,
        'True Label': y_true,
        'Predicted': y_pred,
        'Confidence': confidences,
        'Is Valid Cattle': is_valid_cattle_list
    })
    
    print("\nDetailed Results:")
    print(results_df)
    print("\n")
    
    # Generate confusion matrix
    # Define all possible classes
    classes = ["gir", "murrah", "non_cattle"]
    
    # Ensure all classes are in y_true and y_pred
    all_classes_y_true = [label if label in classes else "non_cattle" for label in y_true]
    all_classes_y_pred = [label if label in classes else "non_cattle" for label in y_pred]
    
    # Create confusion matrix
    cm = confusion_matrix(all_classes_y_true, all_classes_y_pred, labels=classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Cattle Breed Classifier')
    
    # Save confusion matrix
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(all_classes_y_true, all_classes_y_pred, labels=classes))
    
    # Show the plot
    plt.show()
    
    # Return the confusion matrix for further analysis
    return cm, results_df

if __name__ == "__main__":
    generate_confusion_matrix()