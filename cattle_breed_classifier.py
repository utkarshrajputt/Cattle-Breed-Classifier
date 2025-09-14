"""
Cattle Breed Classifier
-----------------------
A proof-of-concept cattle breed classifier using transfer learning.
This script:
1. Loads the image dataset organized in data/train and data/val directories
2. Uses MobileNetV2 for transfer learning
3. Trains the model with data augmentation
4. Evaluates the model and prints accuracy metrics
5. Saves the trained model as 'breed_model.h5'
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# Configuration parameters
IMG_SIZE = 224  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2  # gir and murrah

def create_data_generators():
    """Create train and validation data generators with augmentation."""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Just rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Load images from directories
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'data/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def create_model():
    """Create and compile the model using MobileNetV2 with transfer learning."""
    
    # Load the pre-trained MobileNetV2 model without the top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot the training and validation accuracy and loss."""
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main function to train and evaluate the model."""
    
    print("Preparing data generators...")
    train_generator, validation_generator = create_data_generators()
    
    print("Creating model with MobileNetV2...")
    model = create_model()
    
    # Display model summary
    model.summary()
    
    print("\nTraining the model...")
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"\nValidation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    print("\nSaving the model as 'breed_model.h5'...")
    model.save('breed_model.h5')
    print("Model saved successfully!")
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()