"""
Web application for plant disease detection.
"""

import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from model import PlantDiseaseModel, get_disease_info
from utils import load_and_preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
MODEL_PATH = os.environ.get('MODEL_PATH', 'saved_models/plant_disease_model.h5')


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the trained model."""
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = PlantDiseaseModel()
        model.load(MODEL_PATH)
        print(f"Model loaded from: {MODEL_PATH}")
    return model


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Load model
        current_model = load_model()
        if current_model is None:
            return jsonify({'error': 'Model not found. Please train a model first.'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        image = load_and_preprocess_image(filepath)
        predicted_class, confidence, all_probs = current_model.predict(image)
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        
        # Get top 5 predictions
        top_k_indices = np.argsort(all_probs)[-5:][::-1]
        top_predictions = [
            {
                'plant': get_disease_info(idx)['plant'],
                'disease': get_disease_info(idx)['disease'],
                'probability': float(all_probs[idx] * 100)
            }
            for idx in top_k_indices
        ]
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': {
                'plant': disease_info['plant'],
                'disease': disease_info['disease'],
                'confidence': float(confidence * 100),
                'is_healthy': disease_info['is_healthy']
            },
            'top_predictions': top_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    # Use environment variable to control debug mode
    # Set FLASK_DEBUG=1 in development, leave unset in production
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
