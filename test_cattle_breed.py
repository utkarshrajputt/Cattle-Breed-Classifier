"""
Cattle Breed Classifier - Test Script
------------------------------------
This script loads the trained model and allows testing with new cattle images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'breed_model.h5'

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image for prediction."""
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale just like during training
    return img_array

def predict_breed(model, img_path, confidence_threshold=0.85, entropy_threshold=0.4):
    """Predict the breed of cattle in the image."""
    preprocessed_img = load_and_preprocess_image(img_path)
    
    # Get prediction
    predictions = model.predict(preprocessed_img)
    
    # Get class indices from the model
    class_indices = {}
    for class_dir in os.listdir('data/train'):
        if os.path.isdir(os.path.join('data/train', class_dir)):
            if 'gir' in class_dir.lower():
                class_indices['gir'] = len(class_indices)
            elif 'murrah' in class_dir.lower():
                class_indices['murrah'] = len(class_indices)
    
    # Inverse the class indices to map from index to class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Get the predicted class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = idx_to_class.get(predicted_class_idx, "Unknown")
    confidence = float(predictions[0][predicted_class_idx])
    
    # Calculate prediction entropy (uncertainty measure)
    # Add epsilon to prevent log(0)
    epsilon = 1e-7
    predictions_safe = np.clip(predictions[0], epsilon, 1.0 - epsilon)
    entropy = -np.sum(predictions_safe * np.log2(predictions_safe))
    
    # Normalize entropy (max entropy for n classes is log2(n))
    max_entropy = np.log2(len(class_indices))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate difference between top two predictions if possible
    if len(predictions[0]) >= 2:
        # Get the indices sorted by prediction value
        sorted_indices = np.argsort(-predictions[0])
        if len(sorted_indices) >= 2:
            conf_difference = predictions[0][sorted_indices[0]] - predictions[0][sorted_indices[1]]
        else:
            conf_difference = 1.0
    else:
        conf_difference = 1.0
    
    # Check if the image is likely a cattle breed using multiple criteria
    is_valid_cattle = (confidence >= confidence_threshold and 
                       normalized_entropy <= entropy_threshold and
                       conf_difference >= 0.5)
    
    return predicted_class, confidence, is_valid_cattle, normalized_entropy

def display_prediction(img_path, breed, confidence, is_valid_cattle, entropy=None):
    """Display the image with prediction results."""
    img = plt.imread(img_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    if is_valid_cattle:
        title = f'Predicted: {breed} (Confidence: {confidence:.2f})'
        if entropy is not None:
            title += f'\nEntropy: {entropy:.2f}'
        color = 'green'
    else:
        title = f'⚠️ NOT A RECOGNIZED CATTLE BREED\nPrediction: {breed} (Confidence: {confidence:.2f})'
        if entropy is not None:
            title += f'\nEntropy: {entropy:.2f} (High uncertainty)'
        color = 'red'
        
    plt.title(title, color=color, fontweight='bold')
    plt.axis('off')
    plt.show()

def main():
    """Main function to test the model with new images."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please train the model first using cattle_breed_classifier.py")
        return
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Test directory for new images
    test_dir = 'data/test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory {test_dir} for test images.")
        print(f"Please place test images in the {test_dir} directory and run the script again.")
        return
    
    # Get all image files from test directory
    image_files = [f for f in os.listdir(test_dir) 
                  if os.path.isfile(os.path.join(test_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {test_dir}.")
        print("Please add some .jpg, .jpeg, or .png files to test.")
        return
    
    # Process each image
    print(f"Found {len(image_files)} images for testing.")
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"\nProcessing {img_file}...")
        
        # Predict breed with confidence and entropy thresholds
        breed, confidence, is_valid_cattle, entropy = predict_breed(model, img_path)
        
        if is_valid_cattle:
            print(f"Prediction: {breed} with confidence {confidence:.4f}, entropy: {entropy:.4f}")
        else:
            print(f"⚠️ Warning: The image might not be a cattle breed")
            print(f"Low confidence prediction: {breed} with confidence {confidence:.4f}, entropy: {entropy:.4f}")
        
        display_prediction(img_path, breed, confidence, is_valid_cattle, entropy)

if __name__ == "__main__":
    main()