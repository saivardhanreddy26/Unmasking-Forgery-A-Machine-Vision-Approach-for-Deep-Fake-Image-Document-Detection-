import os
from pathlib import Path

class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    
    # Upload settings
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Model paths
    MODELS_DIR = BASE_DIR / 'saved_models'
    RESNET_MODEL_PATH = MODELS_DIR / 'resnet_cnn.h5'
    ANN_MODEL_PATH = MODELS_DIR / 'ann_classifier.h5'
    GAN_MODEL_PATH = MODELS_DIR / 'gan_generator.h5'
    
    # Dataset paths
    DATA_DIR = BASE_DIR / 'data'
    TRAINING_DIR = DATA_DIR / 'training_real'
    VALIDATION_DIR = DATA_DIR / 'validation_real'
    
    # Model hyperparameters
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # ELA parameters
    ELA_QUALITY = 90
    
    # Create directories if they don't exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
