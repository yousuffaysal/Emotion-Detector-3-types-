import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import model_from_json, load_model
import base64
import sys

# DEBUG LOGGING
print("Starting Backend...", file=sys.stderr)
print(f"Current Working Directory: {os.getcwd()}", file=sys.stderr)

app = Flask(__name__)
CORS(app)

# Global state
current_model = None
current_model_name = "default"
emotion_labels = ["Happy", "Sad", "Neutral"]

def find_available_models():
    """Recursively find all .h5 files in the project"""
    models = []
    # We are in backend/app.py, so root is one level up
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    
    print(f"Searching for models in: {project_root}", file=sys.stderr)
    
    search_dirs = [project_root, backend_dir]
    seen_files = set()

    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            if 'node_modules' in root or '.git' in root or 'venv' in root:
                continue
            for file in files:
                if file.endswith('.h5'):
                    full_path = os.path.join(root, file)
                    if full_path not in seen_files:
                        # Display name
                        models.append({
                            "name": file,
                            "path": full_path,
                            "display": os.path.relpath(full_path, project_root)
                        })
                        seen_files.add(full_path)
    return models

def load_specific_model(path):
    global current_model, current_model_name
    print(f"Attempting to load model from: {path}", file=sys.stderr)
    
    if not os.path.exists(path):
        print("File does not exist!", file=sys.stderr)
        return False
        
    try:
        # Check if there's a matching json file for architecture
        json_path = path.replace('.h5', '.json')
        
        if os.path.exists(json_path):
            print(f"Found architecture JSON: {json_path}", file=sys.stderr)
            with open(json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            current_model = model_from_json(loaded_model_json)
            current_model.load_weights(path)
        else:
            print("No JSON found, loading full model from H5", file=sys.stderr)
            current_model = load_model(path)
            
        current_model_name = os.path.basename(path)
        print(f"Successfully loaded model: {current_model_name}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error loading model {path}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

# Initialize
available_models = find_available_models()
if available_models:
    print(f"Found {len(available_models)} models", file=sys.stderr)
    load_specific_model(available_models[0]['path'])
else:
    print("NO MODELS FOUND!", file=sys.stderr)

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        "current": current_model_name,
        "models": find_available_models()
    })

@app.route('/load_model', methods=['POST'])
def switch_model():
    path = request.json.get('path')
    if load_specific_model(path):
        return jsonify({"success": True, "model": os.path.basename(path)})
    else:
        return jsonify({"error": "Failed to load model"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if current_model is None:
        print("Predict called but NO MODEL LOADED", file=sys.stderr)
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        if 'image_base64' not in request.json:
            return jsonify({"error": "No image provided"}), 400

        img_data = base64.b64decode(request.json['image_base64'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aggressive face detection
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Very loose parameters to force detection
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
        
        face_detected = False
        roi_gray = None

        if len(faces) > 0:
            face_detected = True
            # Use largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
        else:
            return jsonify({
                "face_detected": False,
                "emotion": "N/A",
                "confidence": 0.0,
                "all_scores": {}
            })

        # Preprocess (strictly matching live_emotion_detection.py)
        # Resize to 48x48
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normalize
        roi_gray = roi_gray.astype('float32') / 255.0
        
        # Reshape
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        prediction = current_model.predict(roi_gray, verbose=0)
        
        # DEBUG: Print raw scores
        print(f"Raw scores: {prediction[0]}", file=sys.stderr)
        
        # Get all scores
        scores = {label: float(score) for label, score in zip(emotion_labels, prediction[0])}
        
        max_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][max_index])
        emotion = emotion_labels[max_index] if max_index < len(emotion_labels) else "Unknown"

        return jsonify({
            "face_detected": face_detected,
            "emotion": emotion,
            "confidence": confidence,
            "all_scores": scores
        })

    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 5001 to avoid AirPlay conflict
    app.run(host='0.0.0.0', port=5001, debug=True)
