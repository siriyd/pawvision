import os
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from model import CNN

app = Flask(__name__)

# Model setup
model = CNN()
model_path = os.path.join(os.path.dirname(__file__), 'cats_dogs_model.pth')
model_loaded = False

if os.path.exists(model_path):
    try:
        # Load weights on CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model_loaded = True
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
else:
    print(f"WARNING: Model file '{model_path}' not found. Please run 'train.py' first to train and save the model.")

# Image transforms for prediction (must match validation transformations)
prediction_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Class mapping (corresponds to datasets.ImageFolder default sorting: Cat=0, Dog=1)
CLASSES = {0: 'Cat', 1: 'Dog'}

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    global model_loaded
    if not model_loaded:
        # Try reloading the model in case it was trained since server started
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                model_loaded = True
                print("Model loaded successfully on demand.")
            except Exception as e:
                return jsonify({'error': f"Model file exists but failed to load: {e}"}), 500
        else:
            return jsonify({
                'error': 'Model weights not found. Please run training (train.py) to generate the model weights file first.'
            }), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected for upload'}), 400

    try:
        # Load and transform image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = prediction_transforms(image).unsqueeze(0) # add batch dimension

        # Run prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Format response
        results = {}
        for idx, prob in enumerate(probabilities):
            results[CLASSES[idx]] = float(prob.item() * 100) # percentage
            
        # Get highest probability class
        prediction = max(results, key=results.get)
        confidence = results[prediction]

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': results
        })

    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    # Listen on port 5001 to keep projects separate
    app.run(host='0.0.0.0', port=5001, debug=True)
