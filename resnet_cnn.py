import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class ResNetFeatureExtractor:
    """
    ResNet-50 based CNN for deep feature extraction
    from ELA-processed images
    """
    
    def __init__(self, input_shape=(256, 256, 3), trainable_layers=20):
        self.input_shape = input_shape
        self.trainable_layers = trainable_layers
        self.model = None
    
    def build_model(self, feature_dim=512):
        """
        Build ResNet-50 model with custom top layers
        
        Args:
            feature_dim: Dimension of extracted features
        
        Returns:
            Keras model
        """
        # Input layer
        input_tensor = Input(shape=self.input_shape)
        
        # Load pre-trained ResNet-50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        
        # Freeze early layers, fine-tune deeper layers
        for layer in base_model.layers[:-self.trainable_layers]:
            layer.trainable = False
        
        for layer in base_model.layers[-self.trainable_layers:]:
            layer.trainable = True
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = Dense(1024, activation='relu', 
                 kernel_regularizer=l2(0.01), 
                 name='dense_1024')(x)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        
        x = Dense(feature_dim, activation='relu', 
                 kernel_regularizer=l2(0.01),
                 name=f'dense_{feature_dim}')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        
        # Output features
        features = Dense(feature_dim, activation='relu', 
                        name='feature_output')(x)
        
        self.model = Model(inputs=input_tensor, outputs=features, 
                          name='ResNet50_FeatureExtractor')
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile the model with optimizer and loss
        """
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    
    def extract_features(self, images):
        """
        Extract deep features from images
        
        Args:
            images: Batch of preprocessed images
        
        Returns:
            Feature vectors
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        features = self.model.predict(images, verbose=0)
        return features
    
    def save_model(self, path):
        """Save model to file"""
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        """Load model from file"""
        self.model = tf.keras.models.load_model(path)
