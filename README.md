# PyTorch Cats vs Dogs Web Classifier

An interactive, responsive web application that uses a custom PyTorch Convolutional Neural Network (CNN) to classify uploaded images as either a **Cat** or a **Dog** with real-time confidence scores.

## Features
*   **Custom PyTorch CNN:** Built using 3 Conv blocks (MaxPooling, ReLU) and fully connected layers.
*   **Premium Web UI:** Glassmorphism dashboard styled with modern CSS card grids, neon gradients, and drag-and-drop file upload.
*   **Interactive Visualization:** Real-time prediction display with animated probability charts.
*   **Flask API Backend:** Handles image decoding, PyTorch preprocessing transforms, and model inference on request.

## Project Structure
```
├── app.py           # Flask server controller
├── model.py         # PyTorch CNN model definition
├── train.py         # Training script (trains on training split, evaluates on test split)
├── templates/
│   └── index.html   # Frontend dashboard view
└── static/
    ├── app.js       # Drag-and-drop AJAX uploader
    └── style.css    # Premium glassmorphic stylesheet
```

## Setup & Running

### 1. Install Dependencies
Make sure you have Python, PyTorch, Torchvision, Pillow, and Flask installed:
```bash
pip install torch torchvision pillow flask
```

### 2. Train the Model
The model must be trained on the training set before launching the web server. Run the training script:
```bash
python train.py
```
This will train the custom CNN on the dataset, evaluate it on the test set, and save the weights file to `cats_dogs_model.pth`.

### 3. Start the Web Server
Launch the Flask application:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5001`.
