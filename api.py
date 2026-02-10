"""
🌿 API FastAPI - Détection des Maladies des Plantes
Endpoint pour tester le modèle avec Postman
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import uvicorn

# Initialiser l'application FastAPI
app = FastAPI(
    title="🌿 Plant Disease Detection API",
    description="API pour détecter les maladies des plantes à partir d'images",
    version="1.0.0"
)

# Autoriser CORS pour les tests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "plant_disease_model_complete.pth")
IMAGE_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variables globales pour le modèle
model = None
class_names = None

def load_model():
    """Charger le modèle entraîné"""
    global model, class_names
    
    print("📦 Chargement du modèle...")
    
    # Charger le checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    # Recréer l'architecture du modèle
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé avec {num_classes} classes")
    print(f"📍 Utilisation de: {device}")

# Transformations pour l'image
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    load_model()

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "🌿 Plant Disease Detection API",
        "status": "running",
        "endpoints": {
            "POST /predict": "Envoyer une image pour prédiction",
            "GET /classes": "Liste des classes disponibles",
            "GET /health": "Vérifier l'état de l'API"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifier l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "num_classes": len(class_names) if class_names else 0
    }

@app.get("/classes")
async def get_classes():
    """Retourner la liste des classes"""
    if class_names is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    return {
        "num_classes": len(class_names),
        "classes": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédire la maladie d'une plante à partir d'une image
    
    - **file**: Image de la plante (JPG, PNG)
    
    Retourne:
    - **prediction**: Nom de la classe prédite
    - **confidence**: Confiance en pourcentage
    - **all_probabilities**: Probabilités pour toutes les classes
    """
    
    # Vérifier que le modèle est chargé
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    # Vérifier le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transformer l'image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Préparer la réponse
        predicted_class = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        
        # Top 5 prédictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top5_predictions = [
            {
                "class": class_names[idx.item()],
                "probability": prob.item() * 100
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]
        
        return JSONResponse(content={
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence_percent, 2),
            "top_5_predictions": top5_predictions,
            "message": f"La plante est probablement: {predicted_class}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

if __name__ == "__main__":
    print("🚀 Démarrage de l'API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
