"""
Test script for the cattle breed prediction function
"""

from predict_breed import predict_breed

# Test with the dog image
dog_result = predict_breed('data/test/dog.jpg')
print(f"DOG IMAGE TEST:")
print(f"Is valid cattle: {dog_result['valid_cattle']}")
print(f"Confidence: {dog_result['confidence']:.4f}")
print(f"Prediction ratio: {dog_result['prediction_ratio']:.4f}")
print("-" * 40)

# Test with a cattle image
cattle_result = predict_breed('data/test/Gir1.jpg')
print(f"CATTLE IMAGE TEST:")
print(f"Is valid cattle: {cattle_result['valid_cattle']}")
print(f"Confidence: {cattle_result['confidence']:.4f}")
print(f"Prediction ratio: {cattle_result['prediction_ratio']:.4f}")
print("-" * 40)