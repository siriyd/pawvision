# PyTorch Cats vs Dogs Web Classifier

An interactive, responsive web application that uses a custom PyTorch Convolutional Neural Network (CNN) to classify uploaded images as either a **Cat** or a **Dog** with real-time confidence scores.

## Features
*   **Custom PyTorch CNN:** Built using 3 Conv blocks (MaxPooling, ReLU) and fully connected layers.
*   **Premium Web UI:** Glassmorphism dashboard styled with modern CSS card grids, neon gradients, and drag-and-drop file upload.
*   **Interactive Visualization:** Real-time prediction display with animated probability charts.
*   **Flask API Backend:** Handles image decoding, PyTorch preprocessing transforms, and model inference on request.

## Project Structure
```
├── app.py                 # Flask server controller
├── model.py               # PyTorch CNN model definition
├── train.py               # Training script (trains on training split, evaluates on test split)
├── cats_dogs_model.pth    # Saved PyTorch weights (generated after training)
├── templates/
│   └── index.html         # Frontend dashboard view
└── static/
    ├── app.js             # Drag-and-drop AJAX uploader
    └── style.css          # Premium glassmorphic stylesheet
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

---

## Model & Results

### CNN Architecture
The custom CNN is built from scratch using PyTorch and consists of:
*   **Convolutional Blocks**: 3 convolutional layers (32, 64, and 128 channels) with `ReLU` activations and `2x2 MaxPool` downsampling.
*   **Fully Connected Classifier**: A linear layer mapping feature maps to 256 dimensions, followed by a final linear layer outputting raw logits for the 2 classes (**Cat** vs. **Dog**).

### Training Details & Metrics
The model was trained for **10 epochs** on a CPU with a 70/15/15 split of the dataset.

*   **Training Loss**: ~`0.0225`
*   **Validation Loss**: ~`1.0700`
*   **Final Test Accuracy**: **`81.90%`** (3072 correct predictions out of 3751 test samples)

### Saved Weights (`.pth`)
*   **`cats_dogs_model.pth`**: Contains the state dictionary (learned weights and biases) of the trained CNN model.
*   **Integration**: Upon startup, `app.py` automatically checks for this weights file. If present, it loads the parameters on the CPU (`map_location='cpu'`) and puts the network in evaluation mode (`model.eval()`) for real-time inference.
