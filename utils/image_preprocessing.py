"""
Image preprocessing utilities for plant disease detection.
"""

import numpy as np
from PIL import Image
import os


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image from file and preprocess it for model input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    return img_array


def preprocess_image_array(img_array, target_size=(224, 224)):
    """
    Preprocess a numpy array image.
    
    Args:
        img_array (numpy.ndarray): Image array
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Convert to PIL Image for resizing
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert back to array
    img_array = np.array(img)
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    return img_array


def create_data_generators(train_dir, val_dir, test_dir=None, 
                          batch_size=32, target_size=(224, 224)):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_dir (str): Directory containing training images
        val_dir (str): Directory containing validation images
        test_dir (str): Directory containing test images (optional)
        batch_size (int): Batch size for data generators
        target_size (tuple): Target size for images
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
    
    return train_generator, val_generator, test_generator


def augment_image(img_array):
    """
    Apply random augmentation to an image.
    
    Args:
        img_array (numpy.ndarray): Image array
        
    Returns:
        numpy.ndarray: Augmented image array
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    
    # Generate augmented image
    aug_iter = datagen.flow(img_array, batch_size=1)
    aug_image = next(aug_iter)[0]
    
    return aug_image


def batch_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Preprocess multiple images in batch.
    
    Args:
        image_paths (list): List of image file paths
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Batch of preprocessed images
    """
    images = []
    for path in image_paths:
        img = load_and_preprocess_image(path, target_size)
        images.append(img)
    
    return np.array(images)
