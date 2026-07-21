import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ANNClassifier:
    """
    Artificial Neural Network for classifying forgery features
    """
    
    def __init__(self, input_dim=512):
        self.input_dim = input_dim
        self.model = None
    
    def build_model(self):
        """
        Build fully connected ANN classifier
        
        Returns:
            Keras model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            
            # First hidden layer
            Dense(256, activation='relu', 
                 kernel_regularizer=l2(0.01),
                 name='fc1'),
            BatchNormalization(name='bn1'),
            Dropout(0.5, name='dropout1'),
            
            # Second hidden layer
            Dense(128, activation='relu',
                 kernel_regularizer=l2(0.01),
                 name='fc2'),
            BatchNormalization(name='bn2'),
            Dropout(0.4, name='dropout2'),
            
            # Third hidden layer
            Dense(64, activation='relu',
                 kernel_regularizer=l2(0.01),
                 name='fc3'),
            BatchNormalization(name='bn3'),
            Dropout(0.3, name='dropout3'),
            
            # Fourth hidden layer
            Dense(32, activation='relu',
                 name='fc4'),
            Dropout(0.2, name='dropout4'),
            
            # Output layer (binary classification)
            Dense(1, activation='sigmoid', name='output')
        ], name='ANN_Classifier')
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the ANN model
        """
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
    
    def get_callbacks(self):
        """
        Get training callbacks
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        return [early_stopping, reduce_lr]
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs=50, batch_size=32):
        """
        Train the ANN classifier
        """
        if self.model is None:
            self.compile_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return history
    
    def predict(self, features, threshold=0.5):
        """
        Predict forgery probability
        
        Args:
            features: Input feature vectors
            threshold: Classification threshold
        
        Returns:
            predictions, probabilities
        """
        probabilities = self.model.predict(features, verbose=0)
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities
    
    def save_model(self, path):
        """Save model to file"""
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        """Load model from file"""
        self.model = tf.keras.models.load_model(path)
