"""
Prediction script for plant disease detection.
"""

import os
import argparse
import numpy as np
from model import PlantDiseaseModel, get_disease_info
from utils import load_and_preprocess_image


def predict_disease(model_path, image_path, show_top_k=5):
    """
    Predict plant disease from an image.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image to predict
        show_top_k (int): Number of top predictions to show
        
    Returns:
        dict: Prediction results
    """
    print("=" * 60)
    print("Plant Disease Detection - Prediction")
    print("=" * 60)
    
    # Load model
    print(f"\n[1/3] Loading model from: {model_path}")
    model = PlantDiseaseModel()
    model.load(model_path)
    print("Model loaded successfully!")
    
    # Load and preprocess image
    print(f"\n[2/3] Loading and preprocessing image: {image_path}")
    image = load_and_preprocess_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Make prediction
    print(f"\n[3/3] Making prediction...")
    predicted_class, confidence, all_probs = model.predict(image)
    
    # Get disease information
    disease_info = get_disease_info(predicted_class)
    
    # Print results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"\nPredicted Disease: {disease_info['disease']}")
    print(f"Plant Type: {disease_info['plant']}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Health Status: {'Healthy' if disease_info['is_healthy'] else 'Diseased'}")
    
    # Show top K predictions
    print(f"\n--- Top {show_top_k} Predictions ---")
    top_k_indices = np.argsort(all_probs)[-show_top_k:][::-1]
    
    for i, idx in enumerate(top_k_indices, 1):
        info = get_disease_info(idx)
        prob = all_probs[idx]
        print(f"{i}. {info['plant']} - {info['disease']}: {prob * 100:.2f}%")
    
    return {
        'predicted_class': int(predicted_class),
        'confidence': float(confidence),
        'disease_info': disease_info,
        'top_predictions': [
            {
                'class_index': int(idx),
                'probability': float(all_probs[idx]),
                'info': get_disease_info(idx)
            }
            for idx in top_k_indices
        ]
    }


def batch_predict(model_path, image_dir, output_file=None):
    """
    Predict diseases for all images in a directory.
    
    Args:
        model_path (str): Path to the trained model
        image_dir (str): Directory containing images
        output_file (str): Optional file to save results
        
    Returns:
        list: List of prediction results
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = PlantDiseaseModel()
    model.load(model_path)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"\nFound {len(image_files)} images to process")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        try:
            # Load and preprocess image
            image = load_and_preprocess_image(image_path)
            
            # Predict
            predicted_class, confidence, _ = model.predict(image)
            disease_info = get_disease_info(predicted_class)
            
            result = {
                'filename': image_file,
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'plant': disease_info['plant'],
                'disease': disease_info['disease'],
                'is_healthy': disease_info['is_healthy']
            }
            
            results.append(result)
            
            print(f"  -> {disease_info['plant']} - {disease_info['disease']} ({confidence * 100:.2f}%)")
            
        except Exception as e:
            print(f"  -> Error: {str(e)}")
            results.append({
                'filename': image_file,
                'error': str(e)
            })
    
    # Save results if output file specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main function for prediction."""
    parser = argparse.ArgumentParser(
        description='Predict Plant Disease from Image'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file for prediction'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing multiple images for batch prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch prediction results (JSON format)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model):
        raise ValueError(f"Model file not found: {args.model}")
    
    if args.image and args.image_dir:
        raise ValueError("Please provide either --image or --image-dir, not both")
    
    if not args.image and not args.image_dir:
        raise ValueError("Please provide either --image or --image-dir")
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            raise ValueError(f"Image file not found: {args.image}")
        predict_disease(args.model, args.image, args.top_k)
    
    # Batch prediction
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            raise ValueError(f"Image directory not found: {args.image_dir}")
        batch_predict(args.model, args.image_dir, args.output)


if __name__ == '__main__':
    main()
