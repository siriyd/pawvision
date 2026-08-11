document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    
    // Panel States
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const analysisState = document.getElementById('analysis-state');
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    
    // Result elements
    const predictionValue = document.getElementById('prediction-value');
    const confidenceBadge = document.getElementById('confidence-badge');
    const catText = document.getElementById('prob-cat-text');
    const catFill = document.getElementById('prob-cat-fill');
    const dogText = document.getElementById('prob-dog-text');
    const dogFill = document.getElementById('prob-dog-fill');

    // Drag and Drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleImageUpload(fileInput.files[0]);
        }
    });

    // Make drop zone click trigger file browser
    dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            fileInput.click();
        }
    });

    function showState(state) {
        // Hide all
        placeholderState.classList.add('hidden');
        loadingState.classList.add('hidden');
        analysisState.classList.add('hidden');
        errorState.classList.add('hidden');

        // Show selected
        if (state === 'placeholder') placeholderState.classList.remove('hidden');
        else if (state === 'loading') loadingState.classList.remove('hidden');
        else if (state === 'analysis') analysisState.classList.remove('hidden');
        else if (state === 'error') errorState.classList.remove('hidden');
    }

    function handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file!');
            return;
        }

        // Show preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload and Predict
        showState('loading');
        
        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error') });
            }
            return response.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(error => {
            console.error('Error during prediction:', error);
            errorMessage.textContent = error.message || 'Error occurred during classification.';
            showState('error');
        });
    }

    function renderResults(data) {
        showState('analysis');

        // Set prediction label
        predictionValue.textContent = data.prediction;
        confidenceBadge.textContent = `${data.confidence.toFixed(1)}% Match`;

        // Reset progress bar width first to trigger transition animation
        catFill.style.width = '0%';
        dogFill.style.width = '0%';
        
        // Wait a split second, then set the actual probabilities
        setTimeout(() => {
            const catProb = data.probabilities.Cat;
            const dogProb = data.probabilities.Dog;

            catText.textContent = `${catProb.toFixed(1)}%`;
            catFill.style.width = `${catProb}%`;

            dogText.textContent = `${dogProb.toFixed(1)}%`;
            dogFill.style.width = `${dogProb}%`;
        }, 100);
    }
});
