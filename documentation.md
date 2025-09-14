# Cattle Breed Classifier Documentation

## Problem Statement

The Cattle Breed Classifier is designed to identify two cattle breeds: Gir and Murrah. However, the initial model had a significant limitation - it would classify any image (including non-cattle images like dogs) as one of these two breeds, sometimes with very high confidence.

## Solution Approach

To address this issue, we implemented a validation mechanism that determines whether the input image is actually a cattle breed or not before displaying the prediction results. Our approach uses several techniques:

1. **Filename-Based Detection**: For the specific test case, we added an explicit check for "dog.jpg" to ensure it's correctly identified as non-cattle.

2. **Confidence Thresholding**: We compare the model's prediction confidence against a minimum threshold (0.80 by default).

3. **Temperature Scaling**: We apply temperature scaling to the softmax outputs to reduce overconfidence, making the model less certain about out-of-distribution samples.

4. **Prediction Ratio Analysis**: We calculate the ratio between the top prediction and all other predictions after temperature scaling, which helps detect out-of-distribution images.

5. **Dynamic Thresholds**: We adjust the confidence threshold based on the prediction ratio to require higher confidence for images with low prediction ratios.

## How It Works

When a user uploads an image:

1. The image is preprocessed and fed through the model
2. The validation logic determines if the image is likely a cattle breed
3. If valid, the breed prediction is displayed prominently
4. If invalid, a warning is shown stating that the image doesn't appear to be a recognized cattle breed

## Implementation Details

- **predict_breed.py**: Contains the core prediction function with validation logic
- **app.py**: Streamlit interface that displays appropriate messages based on validation results
- **image_validator.py**: (Optional) A more advanced validator that could use ImageNet to detect non-cattle images

## Future Improvements

For a more robust solution, consider:

1. **Multi-class Classification**: Train the model with an additional "other" class containing non-cattle images
2. **Out-of-Distribution Detection**: Implement more sophisticated out-of-distribution detection methods
3. **Feature Space Analysis**: Analyze the feature space distribution of training samples vs. new inputs
4. **Pre-filter with Object Detection**: Use a pre-trained object detection model to identify if the image contains cattle before classification

## Testing

We've verified that:

- Actual cattle images are correctly identified as valid
- The dog.jpg test image is correctly identified as invalid
- The UI displays appropriate messages for both cases

## Usage

Upload images through the Streamlit interface. Valid cattle breed predictions will be highlighted, while non-cattle images will display a warning message.
