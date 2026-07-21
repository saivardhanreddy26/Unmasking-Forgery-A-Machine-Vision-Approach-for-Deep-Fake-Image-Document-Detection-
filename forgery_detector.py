import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from models.ela_preprocessing import ELAProcessor
from models.resnet_cnn import ResNetFeatureExtractor
from models.ann_classifier import ANNClassifier

class ForgeryDetector:
    """
    Complete end-to-end forgery detection pipeline
    Integrates ELA, ResNet-50 CNN, and ANN classifier
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.ela_processor = ELAProcessor(quality=config.ELA_QUALITY)
        self.cnn_extractor = ResNetFeatureExtractor(
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3)
        )
        self.ann_classifier = ANNClassifier(input_dim=512)
        
        self.is_trained = False

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for forgery detection
        """
        ela_img = self.ela_processor.convert_to_ela_image(
            image_path, 
            output_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        )
        
        # Reshape for model input
        preprocessed = np.array(ela_img, dtype=np.float32) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed

    def detect_forgery(self, image_path, threshold=0.5):
        """
        Detect if an image is forged
        
        Args:
            image_path: Path to input image
            threshold: Classification threshold
        
        Returns:
            Dictionary with detection results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Load trained models first.")
        
        # Preprocess image
        preprocessed = self.preprocess_image(image_path)
        
        # Extract features using CNN
        features = self.cnn_extractor.extract_features(preprocessed)
        
        # Classify using ANN
        prediction, probability = self.ann_classifier.predict(
            features, threshold=threshold
        )
        
        # Generate forensic maps
        forensic_maps = self.ela_processor.generate_forensic_maps(image_path)
        
        # Prepare results
        results = {
            'is_forged': bool(prediction[0][0]),
            'confidence': float(probability[0][0]),
            'forgery_probability': float(probability[0][0]) * 100,
            'authenticity_probability': (1 - float(probability[0][0])) * 100,
            'forensic_maps': forensic_maps,
            'verdict': 'FORGED' if prediction[0][0] == 1 else 'AUTHENTIC'
        }
        
        return results

    def batch_detect(self, image_paths, threshold=0.5):
        """
        Detect forgeries in multiple images
        
        Args:
            image_paths: List of image paths
            threshold: Classification threshold
        
        Returns:
            List of detection results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.detect_forgery(image_path, threshold)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

    def load_models(self, resnet_path, ann_path):
        """
        Load pre-trained models
        
        Args:
            resnet_path: Path to ResNet model
            ann_path: Path to ANN model
        """
        resnet_path = Path(resnet_path)
        ann_path = Path(ann_path)

        if not resnet_path.exists() or not ann_path.exists():
            print("Warning: One or more model files not found. Models will not be loaded.")
            self.is_trained = False
            return

        try:
            self.cnn_extractor.load_model(str(resnet_path))
            self.ann_classifier.load_model(str(ann_path))
            self.is_trained = True
            print("Models loaded successfully!")
        except Exception as e:
            self.is_trained = False
            raise Exception(f"Error loading models: {str(e)}")
    
    def save_models(self, resnet_path, ann_path):
        """
        Save trained models
        """
        self.cnn_extractor.save_model(resnet_path)
        self.ann_classifier.save_model(ann_path)
        print("Models saved successfully!")
