"""
Training script for plant disease detection model.
"""

import os
import argparse
from model import PlantDiseaseModel
from utils import create_data_generators


def train_model(train_dir, val_dir, output_dir='saved_models', 
                epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the plant disease detection model.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        output_dir (str): Directory to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    print("=" * 60)
    print("Plant Disease Detection Model - Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    print("\n[1/4] Loading and preparing data...")
    train_gen, val_gen, _ = create_data_generators(
        train_dir, val_dir, 
        batch_size=batch_size
    )
    
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Create and compile model
    print("\n[2/4] Building model...")
    model = PlantDiseaseModel(num_classes=num_classes)
    model.compile_model(learning_rate=learning_rate)
    model.summary()
    
    # Train model
    print("\n[3/4] Training model...")
    history = model.train(
        train_data=train_gen,
        validation_data=val_gen,
        epochs=epochs
    )
    
    # Save model
    print("\n[4/4] Saving model...")
    model_path = os.path.join(output_dir, 'plant_disease_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    return model, history


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(
        description='Train Plant Disease Detection Model'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        required=True,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        required=True,
        help='Path to validation data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='saved_models',
        help='Directory to save trained model (default: saved_models)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.train_dir):
        raise ValueError(f"Training directory not found: {args.train_dir}")
    if not os.path.exists(args.val_dir):
        raise ValueError(f"Validation directory not found: {args.val_dir}")
    
    # Train model
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()
