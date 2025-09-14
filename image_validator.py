"""
Image Type Validator for Cattle Breed Classifier
-----------------------------------------------
This script implements a simple validator to detect if an image is of a cattle breed or not.
It uses transfer learning with MobileNetV2 and custom thresholds to detect out-of-distribution images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Fixed threshold for animal vs. non-animal classification
CATTLE_THRESHOLD = 0.85  # Minimum confidence required to be considered a cattle
IMAGE_SIZE = (224, 224)  # Image size expected by MobileNetV2

# Load the MobileNetV2 base model without the classification head
base_model = MobileNetV2(weights='imagenet', include_top=True)

def is_cattle_image(image_path):
    """
    Determines if the image contains a cattle breed or not.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    tuple
        (is_cattle, confidence, animal_type) where:
        - is_cattle: boolean indicating if the image likely contains a cattle breed
        - confidence: float representing the confidence in the assessment
        - animal_type: string indicating the type of image detected
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        return False, 0.0, "Image not found"
    
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
        
        # Get the ImageNet predictions
        predictions = base_model.predict(img_array)
        
        # Get the top 5 predictions
        top_indices = np.argsort(-predictions[0])[:5]
        
        # ImageNet classes related to cattle, buffalo, and similar animals
        cattle_related_classes = {
            'ox': 345,
            'water_buffalo': 346,
            'bison': 347,
            'ram': 348,
            'bighorn': 349,
            'ibex': 350,
            'hartebeest': 351,
            'impala': 352,
            'gazelle': 353,
            'bull_mastiff': 243,
            'cow': 344
        }
        
        # Non-animal classes that might be confused with cattle
        non_animal_classes = set(range(400, 1000))  # Objects and non-animals
        
        # Check if any of the top predictions are cattle-related
        cattle_confidence = 0.0
        dog_confidence = 0.0
        animal_confidence = 0.0
        
        # Load ImageNet class names
        with open('imagenet_classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        for i in top_indices:
            pred_class = i
            confidence = float(predictions[0][i])
            
            # Check if it's a cattle-related class
            if pred_class in cattle_related_classes.values():
                cattle_confidence += confidence
            
            # Check if it's a dog class (dogs are in range ~150-270)
            if 150 <= pred_class <= 270:
                dog_confidence += confidence
            
            # Check if it's an animal class (roughly the first 400 classes in ImageNet)
            if pred_class < 400:
                animal_confidence += confidence
        
        # Determine the image type
        is_cattle = cattle_confidence > CATTLE_THRESHOLD
        is_dog = dog_confidence > 0.5
        is_animal = animal_confidence > 0.7
        
        if is_cattle:
            return True, cattle_confidence, "cattle"
        elif is_dog:
            return False, dog_confidence, "dog"
        elif is_animal:
            return False, animal_confidence, "other_animal"
        else:
            return False, 1.0 - animal_confidence, "non_animal"
    
    except Exception as e:
        print(f"Error in cattle validation: {str(e)}")
        return False, 0.0, "error"

if __name__ == "__main__":
    # Download the ImageNet class names if not available
    if not os.path.exists('imagenet_classes.txt'):
        # Create a simple imagenet classes file with a few relevant entries
        with open('imagenet_classes.txt', 'w') as f:
            for i in range(1000):
                if i == 243:
                    f.write("bull_mastiff\n")
                elif i == 344:
                    f.write("ox\n")
                elif i == 345:
                    f.write("cow\n")
                elif i == 346:
                    f.write("water_buffalo\n")
                elif i == 347:
                    f.write("bison\n")
                else:
                    f.write(f"class_{i}\n")
    
    # Test directory
    test_dir = 'data/test'
    if os.path.exists(test_dir):
        # Process all images in the test directory
        for file_name in os.listdir(test_dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(test_dir, file_name)
                is_cattle, confidence, animal_type = is_cattle_image(file_path)
                
                result = "CATTLE" if is_cattle else "NOT CATTLE"
                print(f"Image: {file_name}")
                print(f"Result: {result} (Type: {animal_type}, Confidence: {confidence:.4f})")
                print("-" * 40)