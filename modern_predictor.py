"""
Modern Cattle Breed Prediction
------------------------------
Updated prediction script for the modern multi-class cattle breed classifier.
Supports all 41 cattle breeds with confidence scoring and visualization.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
import json

class ModernCattlePredictor:
    def __init__(self, model_path='best_modern_model.pth', results_path='modern_training_results.json'):
        """
        Initialize the modern cattle breed predictor
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        results_path : str
            Path to the training results JSON file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes = None
        self.config = None
        self.transform = None
        
        # Load model and configuration
        self.load_model(model_path, results_path)
        self.setup_transforms()
        
    def load_model(self, model_path, results_path):
        """Load the trained model and configuration"""
        try:
            # Load training results to get classes and config
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            self.classes = results['classes']
            self.config = results['config']
            self.best_accuracy = results.get('best_accuracy', 0.0)  # Store best accuracy
            num_classes = len(self.classes)
            
            print(f"📋 Loaded configuration for {num_classes} classes")
            print(f"   Classes: {', '.join(self.classes[:5])}{'...' if len(self.classes) > 5 else ''}")
            print(f"   Best accuracy: {self.best_accuracy:.2f}%")
            
            # Create model architecture
            self.model = timm.create_model(
                self.config['model_name'],
                pretrained=False,  # We'll load our trained weights
                num_classes=num_classes,
                drop_path_rate=self.config.get('drop_path_rate', 0.2)
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e}")
            print("Make sure you've trained the model first using modern_cattle_classifier.py")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        img_size = self.config.get('img_size', 224)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        torch.Tensor
            Preprocessed image tensor
        PIL.Image
            Original PIL image for display
        """
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor.to(self.device), image
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_single(self, image_path, top_k=5, confidence_threshold=0.1):
        """
        Predict cattle breed for a single image
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        top_k : int
            Number of top predictions to return
        confidence_threshold : float
            Minimum confidence threshold for predictions
            
        Returns:
        --------
        dict
            Prediction results with probabilities and breed names
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        # Format results
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            if prob >= confidence_threshold:
                predictions.append({
                    'rank': i + 1,
                    'breed': self.classes[idx],
                    'confidence': float(prob),
                    'percentage': float(prob * 100)
                })
        
        # Overall confidence (max probability)
        max_confidence = float(top_probs[0])
        predicted_breed = self.classes[top_indices[0]]
        
        return {
            'image_path': image_path,
            'predicted_breed': predicted_breed,
            'confidence': max_confidence,
            'percentage': max_confidence * 100,
            'top_predictions': predictions,
            'original_image': original_image
        }
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict cattle breeds for multiple images
        
        Parameters:
        -----------
        image_paths : list
            List of image file paths
        top_k : int
            Number of top predictions per image
            
        Returns:
        --------
        list
            List of prediction results
        """
        results = []
        
        print(f"🔍 Predicting breeds for {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"   Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
                result = self.predict_single(image_path, top_k)
                results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, result, save_path=None):
        """
        Visualize prediction results
        
        Parameters:
        -----------
        result : dict
            Prediction result from predict_single()
        save_path : str, optional
            Path to save the visualization
        """
        if 'error' in result:
            print(f"Cannot visualize: {result['error']}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(result['original_image'])
        ax1.set_title(f"Input Image\n{os.path.basename(result['image_path'])}", fontsize=12)
        ax1.axis('off')
        
        # Display top predictions
        breeds = [pred['breed'] for pred in result['top_predictions']]
        confidences = [pred['percentage'] for pred in result['top_predictions']]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(breeds))
        bars = ax2.barh(y_pos, confidences, color='skyblue', alpha=0.8)
        
        # Highlight the top prediction
        if bars:
            bars[0].set_color('lightgreen')
            bars[0].set_alpha(1.0)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(breeds)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title(f'Top Predictions\nPredicted: {result["predicted_breed"]} ({result["percentage"]:.1f}%)', fontsize=12)
        ax2.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def get_breed_info(self, breed_name):
        """
        Get information about a cattle breed
        (This is a placeholder - you could extend this with a breed database)
        
        Parameters:
        -----------
        breed_name : str
            Name of the cattle breed
            
        Returns:
        --------
        dict
            Basic breed information
        """
        # This is a simplified example - you could extend this with real breed data
        breed_info = {
            'name': breed_name,
            'origin': 'Information not available',
            'characteristics': 'Information not available',
            'purpose': 'Information not available'
        }
        
        # Add some basic information for common breeds
        if 'gir' in breed_name.lower():
            breed_info.update({
                'origin': 'India (Gujarat)',
                'characteristics': 'Heat resistant, good milk production',
                'purpose': 'Dairy'
            })
        elif 'murrah' in breed_name.lower():
            breed_info.update({
                'origin': 'India (Punjab, Haryana)',
                'characteristics': 'High milk yield, black color',
                'purpose': 'Dairy (Buffalo)'
            })
        elif 'jersey' in breed_name.lower():
            breed_info.update({
                'origin': 'Jersey Island',
                'characteristics': 'Small size, high butterfat content',
                'purpose': 'Dairy'
            })
        elif 'holstein' in breed_name.lower():
            breed_info.update({
                'origin': 'Netherlands',
                'characteristics': 'Large size, high milk production',
                'purpose': 'Dairy'
            })
        
        return breed_info

def main():
    """Demo function to show how to use the predictor"""
    print("🐄 Modern Cattle Breed Predictor")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = ModernCattlePredictor()
        
        # Test with sample images
        test_images = []
        test_dir = 'data/test'
        
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(test_dir, file))
        
        if not test_images:
            print("⚠️ No test images found in data/test/")
            print("Please add some cattle images to test the predictor.")
            return
        
        print(f"Found {len(test_images)} test images")
        
        # Make predictions
        for image_path in test_images[:3]:  # Test first 3 images
            print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
            print("-" * 30)
            
            result = predictor.predict_single(image_path, top_k=5)
            
            print(f"🏆 Predicted Breed: {result['predicted_breed']}")
            print(f"📊 Confidence: {result['percentage']:.1f}%")
            print(f"\nTop 5 Predictions:")
            
            for pred in result['top_predictions']:
                print(f"   {pred['rank']}. {pred['breed']}: {pred['percentage']:.1f}%")
            
            # Show breed info
            breed_info = predictor.get_breed_info(result['predicted_breed'])
            print(f"\n📝 Breed Information:")
            print(f"   Origin: {breed_info['origin']}")
            print(f"   Purpose: {breed_info['purpose']}")
            
            # Visualize results
            save_name = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            predictor.visualize_prediction(result, save_name)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model using modern_cattle_classifier.py")
        print("2. The files best_modern_model.pth and modern_training_results.json exist")
        print("3. Added test images to data/test/ directory")

if __name__ == "__main__":
    main()