"""
Example usage of the Plant Disease Detection Model.

This script demonstrates how to use the model programmatically.
"""

import numpy as np
from model import PlantDiseaseModel, DISEASE_CLASSES, get_disease_info


def example_build_and_compile():
    """Example: Build and compile a new model."""
    print("=" * 60)
    print("Example 1: Building and Compiling a Model")
    print("=" * 60)
    
    # Create model instance
    model = PlantDiseaseModel(num_classes=38, input_shape=(224, 224, 3))
    
    # Build the model
    model.build_model()
    
    # Compile the model
    model.compile_model(learning_rate=0.001)
    
    # Show model summary
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel built and compiled successfully!")


def example_disease_classes():
    """Example: Display all supported disease classes."""
    print("\n" + "=" * 60)
    print("Example 2: Supported Disease Classes")
    print("=" * 60)
    
    print(f"\nTotal number of classes: {len(DISEASE_CLASSES)}\n")
    
    # Group by plant type
    plant_diseases = {}
    for disease in DISEASE_CLASSES:
        parts = disease.split('___')
        plant = parts[0] if len(parts) > 0 else 'Unknown'
        disease_name = parts[1] if len(parts) > 1 else 'Unknown'
        
        if plant not in plant_diseases:
            plant_diseases[plant] = []
        plant_diseases[plant].append(disease_name)
    
    # Display grouped diseases
    for plant, diseases in sorted(plant_diseases.items()):
        print(f"\n{plant}:")
        for disease in diseases:
            status = "✓" if disease.lower() == 'healthy' else "✗"
            print(f"  {status} {disease}")


def example_disease_info():
    """Example: Get information about specific diseases."""
    print("\n" + "=" * 60)
    print("Example 3: Disease Information")
    print("=" * 60)
    
    # Get info for a few example classes
    example_indices = [0, 10, 20, 30, 37]
    
    for idx in example_indices:
        info = get_disease_info(idx)
        print(f"\nClass {idx}:")
        print(f"  Plant: {info['plant']}")
        print(f"  Disease: {info['disease']}")
        print(f"  Healthy: {info['is_healthy']}")
        print(f"  Full Name: {info['full_name']}")


def example_model_prediction_simulation():
    """Example: Simulate a prediction (without actual image)."""
    print("\n" + "=" * 60)
    print("Example 4: Simulated Prediction")
    print("=" * 60)
    
    print("\nNote: This is a simulation. For real predictions, use predict.py")
    print("      with actual plant images.\n")
    
    # Simulate prediction probabilities
    num_classes = len(DISEASE_CLASSES)
    simulated_probs = np.random.dirichlet(np.ones(num_classes), size=1)[0]
    
    # Get top 5 predictions
    top_k_indices = np.argsort(simulated_probs)[-5:][::-1]
    
    print("Top 5 Predicted Diseases (simulated):")
    for i, idx in enumerate(top_k_indices, 1):
        info = get_disease_info(idx)
        prob = simulated_probs[idx]
        print(f"{i}. {info['plant']} - {info['disease']}: {prob * 100:.2f}%")


def example_model_architecture_details():
    """Example: Display model architecture details."""
    print("\n" + "=" * 60)
    print("Example 5: Model Architecture Details")
    print("=" * 60)
    
    print("""
Model Architecture:
    
Input Layer:
    - Shape: (224, 224, 3) - RGB images
    
Convolutional Block 1:
    - Conv2D: 32 filters, 3x3 kernel
    - BatchNormalization
    - Conv2D: 32 filters, 3x3 kernel
    - BatchNormalization
    - MaxPooling2D: 2x2
    - Dropout: 25%
    
Convolutional Block 2:
    - Conv2D: 64 filters, 3x3 kernel
    - BatchNormalization
    - Conv2D: 64 filters, 3x3 kernel
    - BatchNormalization
    - MaxPooling2D: 2x2
    - Dropout: 25%
    
Convolutional Block 3:
    - Conv2D: 128 filters, 3x3 kernel
    - BatchNormalization
    - Conv2D: 128 filters, 3x3 kernel
    - BatchNormalization
    - MaxPooling2D: 2x2
    - Dropout: 25%
    
Convolutional Block 4:
    - Conv2D: 256 filters, 3x3 kernel
    - BatchNormalization
    - Conv2D: 256 filters, 3x3 kernel
    - BatchNormalization
    - MaxPooling2D: 2x2
    - Dropout: 25%
    
Fully Connected Layers:
    - Flatten
    - Dense: 512 units, ReLU activation
    - BatchNormalization
    - Dropout: 50%
    - Dense: 256 units, ReLU activation
    - BatchNormalization
    - Dropout: 50%
    
Output Layer:
    - Dense: 38 units, Softmax activation
    
Total Parameters: ~10M (approximate)
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PLANT DISEASE DETECTION MODEL - EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_build_and_compile()
    example_disease_classes()
    example_disease_info()
    example_model_prediction_simulation()
    example_model_architecture_details()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Prepare your dataset in the correct format")
    print("2. Run 'python train.py' to train the model")
    print("3. Run 'python predict.py' to make predictions")
    print("4. Run 'python app.py' to start the web interface")
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
