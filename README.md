# 🌿 Plant Disease Detection Model

A deep learning-based system for detecting diseases in plant leaves using Convolutional Neural Networks (CNN). This model can identify 38 different plant disease classes across multiple plant species.

## 📋 Overview

This project implements a CNN-based image classification model that can detect various plant diseases from leaf images. It includes:

- **Deep Learning Model**: CNN architecture built with TensorFlow/Keras
- **38 Disease Classes**: Supports multiple plant types including Tomato, Potato, Apple, Grape, Corn, and more
- **Web Interface**: User-friendly Flask web application for easy disease detection
- **CLI Tools**: Command-line scripts for training and prediction
- **Data Augmentation**: Built-in image preprocessing and augmentation utilities

## 🎯 Supported Plant Diseases

The model can detect diseases in the following plants:

- **Apple**: Scab, Black rot, Cedar apple rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Powdery mildew, Healthy
- **Corn (Maize)**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Orange**: Haunglongbing (Citrus greening)
- **Peach**: Bacterial spot, Healthy
- **Pepper (Bell)**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery mildew
- **Strawberry**: Leaf scorch, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/team202300/PLANT-DISEASE-DETECTION-MODEL.git
cd PLANT-DISEASE-DETECTION-MODEL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Web Application

Launch the web interface for interactive disease detection:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

The web interface allows you to:
- Upload plant leaf images
- Get instant disease predictions
- View confidence scores and top predictions
- See health status of the plant

### 2. Command-Line Prediction

Predict disease for a single image:

```bash
python predict.py --model saved_models/plant_disease_model.h5 --image path/to/leaf.jpg
```

Batch prediction for multiple images:

```bash
python predict.py --model saved_models/plant_disease_model.h5 --image-dir path/to/images/ --output results.json
```

### 3. Training a New Model

To train the model on your own dataset:

```bash
python train.py --train-dir path/to/train --val-dir path/to/validation --epochs 50 --batch-size 32
```

**Training Arguments:**
- `--train-dir`: Path to training data directory
- `--val-dir`: Path to validation data directory
- `--output-dir`: Directory to save trained model (default: `saved_models`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)

**Dataset Structure:**
```
train/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
│   └── ...
└── ...

validation/
├── Apple___Apple_scab/
│   └── ...
└── ...
```

## 🏗️ Model Architecture

The model uses a deep CNN architecture with:

- **4 Convolutional Blocks**: Each with double Conv2D layers, BatchNormalization, MaxPooling, and Dropout
- **Feature Maps**: Progressive increase (32 → 64 → 128 → 256)
- **Fully Connected Layers**: Two dense layers (512 and 256 units) with dropout
- **Output Layer**: Softmax activation for multi-class classification
- **Regularization**: Batch normalization and dropout for preventing overfitting
- **Data Augmentation**: Random rotation, shifts, zoom, and flips during training

**Input**: 224×224 RGB images  
**Output**: Probability distribution over 38 disease classes

## 📦 Project Structure

```
PLANT-DISEASE-DETECTION-MODEL/
├── model/
│   ├── __init__.py
│   └── plant_disease_model.py    # CNN model implementation
├── utils/
│   ├── __init__.py
│   └── image_preprocessing.py    # Image processing utilities
├── templates/
│   └── index.html                # Web interface template
├── app.py                        # Flask web application
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## 🔧 Dependencies

- **tensorflow** >= 2.10.0: Deep learning framework
- **numpy** >= 1.21.0: Numerical computing
- **pillow** >= 9.0.0: Image processing
- **matplotlib** >= 3.5.0: Visualization
- **scikit-learn** >= 1.0.0: Machine learning utilities
- **flask** >= 2.0.0: Web framework

## 💡 Usage Examples

### Python API

```python
from model import PlantDiseaseModel, get_disease_info
from utils import load_and_preprocess_image

# Load the model
model = PlantDiseaseModel()
model.load('saved_models/plant_disease_model.h5')

# Load and preprocess an image
image = load_and_preprocess_image('path/to/leaf.jpg')

# Make prediction
predicted_class, confidence, probabilities = model.predict(image)

# Get disease information
disease_info = get_disease_info(predicted_class)

print(f"Plant: {disease_info['plant']}")
print(f"Disease: {disease_info['disease']}")
print(f"Confidence: {confidence * 100:.2f}%")
print(f"Healthy: {disease_info['is_healthy']}")
```

### Building a Custom Model

```python
from model import PlantDiseaseModel

# Create model with custom parameters
model = PlantDiseaseModel(num_classes=38, input_shape=(224, 224, 3))

# Build and compile
model.build_model()
model.compile_model(learning_rate=0.001)

# View architecture
model.summary()

# Train
history = model.train(
    train_data=train_generator,
    validation_data=val_generator,
    epochs=50,
    batch_size=32
)

# Save
model.save('my_model.h5')
```

## 🎓 Training Tips

1. **Dataset**: Use the PlantVillage dataset or similar plant disease datasets
2. **Data Augmentation**: Already implemented in the training pipeline
3. **Early Stopping**: Automatically stops training when validation loss stops improving
4. **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
5. **GPU Acceleration**: Automatically uses GPU if available with TensorFlow

## 📊 Model Performance

The model achieves:
- High accuracy on PlantVillage dataset
- Robust performance across different plant species
- Good generalization with data augmentation
- Real-time inference capability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- PlantVillage dataset for providing training data
- TensorFlow/Keras for the deep learning framework
- The open-source community for various tools and libraries

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for agriculture and plant health**
