"""
Plant Disease Detection Model
A CNN-based deep learning model for detecting diseases in plant leaves.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
import numpy as np


class PlantDiseaseModel:
    """
    Convolutional Neural Network model for plant disease detection.
    
    This model is designed to classify plant leaf images into various disease categories.
    """
    
    def __init__(self, num_classes=38, input_shape=(224, 224, 3)):
        """
        Initialize the plant disease detection model.
        
        Args:
            num_classes (int): Number of disease classes to detect (default: 38)
            input_shape (tuple): Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """
        Build the CNN architecture for plant disease detection.
        
        Returns:
            keras.Model: Compiled model ready for training
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate (float): Learning rate for the optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model on provided data.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of Keras callbacks
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            self.compile_model()
            
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """
        Predict disease class for a single image.
        
        Args:
            image (numpy.ndarray): Preprocessed image array
            
        Returns:
            tuple: (predicted_class_index, confidence_score, all_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Please build/load the model first.")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        return predicted_class, confidence, predictions[0]
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please build the model first.")
        self.model.save(filepath)
        
    def load(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Please build/load the model first.")
        
        results = self.model.evaluate(test_data, return_dict=True)
        return results


# Disease class labels for PlantVillage dataset
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def get_disease_info(class_index):
    """
    Get disease information for a predicted class.
    
    Args:
        class_index (int): Index of the predicted class
        
    Returns:
        dict: Dictionary containing disease information
    """
    if class_index < 0 or class_index >= len(DISEASE_CLASSES):
        return {"error": "Invalid class index"}
    
    disease_name = DISEASE_CLASSES[class_index]
    parts = disease_name.split('___')
    
    return {
        'class_index': class_index,
        'full_name': disease_name,
        'plant': parts[0] if len(parts) > 0 else 'Unknown',
        'disease': parts[1] if len(parts) > 1 else 'Unknown',
        'is_healthy': 'healthy' in disease_name.lower()
    }
