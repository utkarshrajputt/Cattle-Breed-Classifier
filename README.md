# Cattle Breed Classifier

A proof-of-concept cattle breed classifier built with TensorFlow and Keras that distinguishes between Gir and Murrah cattle breeds.

## Overview

This project uses transfer learning with pre-trained MobileNetV2 architecture to classify cattle breeds. The model is trained to distinguish between two breeds:

- Gir
- Murrah

## Project Structure

```
Avishkar Project/
├── data/
│   ├── train/
│   │   ├── gir/          # Training images of Gir breed
│   │   └── murrah/       # Training images of Murrah breed
│   ├── val/
│   │   ├── gir/          # Validation images of Gir breed
│   │   └── murrah/       # Validation images of Murrah breed
│   └── test/             # For testing new images
├── venv/                  # Python virtual environment
├── temp/                  # Temporary storage for uploaded images
├── records/               # CSV storage for prediction records
├── cattle_breed_classifier.py  # Main training script
├── test_cattle_breed.py        # Script for testing the model
├── predict_breed.py            # Prediction function for single images
├── app.py                      # Streamlit web application
├── breed_model.h5              # Saved model (after training)
└── training_history.png        # Training/validation performance graphs
```

## Setup and Requirements

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Installation

1. Clone this repository or download the source code
2. Create and activate a virtual environment (optional but recommended)

   ```powershell
   # Create virtual environment
   python -m venv venv

   # Activate the environment
   .\venv\Scripts\Activate.ps1
   ```

3. Install required packages
   ```powershell
   pip install tensorflow numpy matplotlib pillow streamlit pandas
   ```

## Data Preparation

1. Place your training images in the appropriate directories:

   - `data/train/gir/` - For Gir breed training images
   - `data/train/murrah/` - For Murrah breed training images
   - `data/val/gir/` - For Gir breed validation images
   - `data/val/murrah/` - For Murrah breed validation images

2. Make sure the images are in formats like JPG, JPEG, or PNG

## Usage

### Training the Model

Run the training script:

```powershell
python cattle_breed_classifier.py
```

This will:

1. Load and preprocess the images
2. Create a model using MobileNetV2 architecture
3. Train the model with data augmentation
4. Evaluate on the validation set
5. Save the trained model as `breed_model.h5`
6. Generate and save a plot of training metrics as `training_history.png`

### Testing the Model

1. Place test images in the `data/test/` directory
2. Run the testing script:
   ```powershell
   python test_cattle_breed.py
   ```
3. The script will display each image along with the predicted breed and confidence score

### Using the Web Application

1. Make sure you have trained the model and have `breed_model.h5` file
2. Launch the Streamlit app:
   ```powershell
   python -m streamlit run app.py
   ```
3. Access the app in your web browser (typically at http://localhost:8501)
4. Upload an image of a cattle
5. View the predictions and confidence scores
6. Optionally save the results to a CSV file using the "Save Record" button

## Model Details

- Base model: MobileNetV2 (pre-trained on ImageNet)
- Input image size: 224×224 pixels
- Data augmentation: rotation, shift, shear, zoom, flip
- Training parameters:
  - Optimizer: Adam
  - Loss function: Categorical Cross-Entropy
  - Batch size: 32
  - Epochs: 20

## Improving the Model

To improve the model's performance:

1. Increase the dataset size with more diverse images
2. Try different transfer learning architectures (e.g., EfficientNet)
3. Apply fine-tuning to the base model after initial training
4. Experiment with different hyperparameters

## Web Application Features

The Streamlit web application (`app.py`) provides:

- User-friendly interface for cattle breed identification
- Image upload capability (supports JPG, JPEG, PNG formats)
- Display of top 2 predicted breeds with confidence scores
- Visual representation of prediction confidence with bar chart
- "Save Record" functionality that stores predictions in a CSV file (mock BPA integration)
- Record viewing capability to see previously saved predictions

## License

[Specify the license or indicate if the project is for educational purposes only]
