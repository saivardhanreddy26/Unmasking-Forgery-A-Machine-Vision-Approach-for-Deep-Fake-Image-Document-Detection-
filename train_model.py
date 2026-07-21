import os
import sys

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# IMPORTANT: Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Fix protobuf issue
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Disable GPU for Mac M1/M2/M3 if causing issues
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import TensorFlow AFTER setting environment variables
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Configure TensorFlow for Mac
try:
    # For Apple Silicon Macs - use CPU only to avoid Metal issues
    tf.config.set_visible_devices([], 'GPU')
    print("Running on CPU only (Metal GPU disabled)")
except:
    pass

from config import Config
from models.ela_preprocessing import ELAProcessor
from models.resnet_cnn import ResNetFeatureExtractor
from models.ann_classifier import ANNClassifier

class ForgeryDetectorTrainer:
    """
    Training pipeline for the forgery detection system
    """
    
    def __init__(self, config):
        self.config = config
        self.ela_processor = ELAProcessor(quality=config.ELA_QUALITY)
        self.cnn_extractor = ResNetFeatureExtractor(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        )
        self.ann_classifier = ANNClassifier(input_dim=512)
    
    def load_dataset(self, dataset_path):
        """
        Load and preprocess the dataset
        """
        print("Loading dataset...")
        
        images = []
        labels = []
        
        # Load real images (label 0)
        real_path = Path(dataset_path) / 'real'
        print(f"Looking for real images in: {real_path}")
        
        if real_path.exists():
            image_files = list(real_path.glob('*.jpg')) + list(real_path.glob('*.png'))
            print(f"Found {len(image_files)} real images")
            
            for idx, img_file in enumerate(image_files[:500]):  # Limit for faster testing
                try:
                    ela_img = self.ela_processor.convert_to_ela_image(
                        str(img_file),
                        output_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
                    )
                    images.append(ela_img)
                    labels.append(0)  # Real
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} real images...")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        else:
            print(f"Warning: Real images path does not exist: {real_path}")
        
        # Load fake images (label 1)
        fake_path = Path(dataset_path) / 'fake'
        print(f"Looking for fake images in: {fake_path}")
        
        if fake_path.exists():
            image_files = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))
            print(f"Found {len(image_files)} fake images")
            
            for idx, img_file in enumerate(image_files[:500]):  # Limit for faster testing
                try:
                    ela_img = self.ela_processor.convert_to_ela_image(
                        str(img_file),
                        output_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
                    )
                    images.append(ela_img)
                    labels.append(1)  # Fake
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} fake images...")
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        else:
            print(f"Warning: Fake images path does not exist: {fake_path}")
        
        if len(images) == 0:
            raise ValueError("No images loaded! Check your dataset path.")
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(images)}")
        print(f"Real: {np.sum(labels == 0)}, Fake: {np.sum(labels == 1)}")
        print(f"Images shape: {images.shape}")
        
        return images, labels
    
    def train_pipeline(self, X_train, y_train, X_val, y_val):
        """
        Train the complete CNN-ANN pipeline
        """
        print("\n" + "="*60)
        print("TRAINING RESNET-50 FEATURE EXTRACTOR")
        print("="*60)
        
        # Build CNN
        print("\nBuilding ResNet-50 model...")
        self.cnn_extractor.build_model(feature_dim=512)
        print("Model built successfully!")
        
        print("\nCompiling model...")
        self.cnn_extractor.compile_model(
            learning_rate=self.config.LEARNING_RATE
        )
        print("Model compiled!")
        
        # Extract features (batch processing to avoid memory issues)
        print("\nExtracting features from training set...")
        batch_size = 32
        train_features_list = []
        
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]
            batch_features = self.cnn_extractor.extract_features(batch)
            train_features_list.append(batch_features)
            print(f"Processed {min(i+batch_size, len(X_train))}/{len(X_train)} training samples")
        
        train_features = np.vstack(train_features_list)
        
        print("\nExtracting features from validation set...")
        val_features_list = []
        
        for i in range(0, len(X_val), batch_size):
            batch = X_val[i:i+batch_size]
            batch_features = self.cnn_extractor.extract_features(batch)
            val_features_list.append(batch_features)
            print(f"Processed {min(i+batch_size, len(X_val))}/{len(X_val)} validation samples")
        
        val_features = np.vstack(val_features_list)
        
        print(f"\nTraining features shape: {train_features.shape}")
        print(f"Validation features shape: {val_features.shape}")
        
        # Train ANN Classifier
        print("\n" + "="*60)
        print("TRAINING ANN CLASSIFIER")
        print("="*60)
        
        self.ann_classifier.build_model()
        self.ann_classifier.compile_model(learning_rate=0.001)
        
        history = self.ann_classifier.train(
            train_features, y_train,
            val_features, y_val,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Extract features in batches
        batch_size = 32
        test_features_list = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_features = self.cnn_extractor.extract_features(batch)
            test_features_list.append(batch_features)
            print(f"Processed {min(i+batch_size, len(X_test))}/{len(X_test)} test samples")
        
        test_features = np.vstack(test_features_list)
        
        # Get predictions
        predictions, probabilities = self.ann_classifier.predict(test_features)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                   target_names=['Real', 'Fake']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        return predictions, probabilities
    
    def plot_training_history(self, history):
        """
        Plot training curves
        """
        print("\nGenerating training history plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2)
            axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision/Recall
        if 'precision' in history.history:
            axes[1, 1].plot(history.history['precision'], label='Precision', linewidth=2)
            axes[1, 1].plot(history.history['recall'], label='Recall', linewidth=2)
            axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Training history saved as 'training_history.png'")
    
    def save_models(self):
        """
        Save trained models
        """
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)
        
        self.cnn_extractor.save_model(str(self.config.RESNET_MODEL_PATH))
        print(f"ResNet model saved: {self.config.RESNET_MODEL_PATH}")
        
        self.ann_classifier.save_model(str(self.config.ANN_MODEL_PATH))
        print(f"ANN model saved: {self.config.ANN_MODEL_PATH}")
        
        print("\nModels saved successfully!")

def main():
    """
    Main training function
    """
    print("="*60)
    print("IMAGE FORGERY DETECTION - TRAINING PIPELINE")
    print("="*60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Python Version: {sys.version}")
    print("="*60)
    
    config = Config()
    trainer = ForgeryDetectorTrainer(config)
    
    # Load dataset
    dataset_path = config.TRAINING_DIR
    print(f"\nDataset path: {dataset_path}")
    
    try:
        X, y = trainer.load_dataset(dataset_path)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nPlease ensure:")
        print("1. Dataset is downloaded from Kaggle")
        print("2. Images are organized as:")
        print(f"   - {dataset_path}/real/*.jpg")
        print(f"   - {dataset_path}/fake/*.jpg")
        sys.exit(1)
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train pipeline
    try:
        history = trainer.train_pipeline(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate
    predictions, probabilities = trainer.evaluate_model(X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Models saved in: {config.MODELS_DIR}")
    print("You can now run the Flask app: python app.py")

if __name__ == "__main__":
    main()
