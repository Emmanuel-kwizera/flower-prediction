const API_URL = 'https://flower-prediction-app-3c3y.onrender.com';

// Elements
const apiStatus = document.getElementById('api-status');
const modelStatus = document.getElementById('model-status');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');
const resultArea = document.getElementById('result-area');
const predictionClass = document.getElementById('prediction-class');
const confidenceBar = document.getElementById('confidence-bar');
const predictionConfidence = document.getElementById('prediction-confidence');
const retrainBtn = document.getElementById('retrain-btn');
const trainMsg = document.getElementById('train-msg');

// 1. Check API Status
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            apiStatus.textContent = 'API Online';
            apiStatus.className = 'status-badge online';
            modelStatus.textContent = data.model_status;
        } else {
            throw new Error('API Error');
        }
    } catch (error) {
        apiStatus.textContent = 'API Offline';
        apiStatus.className = 'status-badge offline';
        modelStatus.textContent = 'Unreachable';
    }
}

// Initial check and poll every 10s
checkStatus();
setInterval(checkStatus, 10000);

// 2. Image Upload Handling
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#475569';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#475569';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        dropZone.querySelector('p').hidden = true;
        predictBtn.disabled = false;
        resultArea.hidden = true;
    };
    reader.readAsDataURL(file);
    fileInput.file = file; // Attach file object for upload
}

// 3. Prediction
predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0] || fileInput.file;
    if (!file) return;

    predictBtn.textContent = 'Analyzing...';
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            showResult(data);
        } else {
            alert('Prediction failed. Ensure model is loaded.');
        }
    } catch (error) {
        console.error(error);
        alert('Error connecting to API.');
    } finally {
        predictBtn.textContent = 'Analyze Flower';
        predictBtn.disabled = false;
    }
});

function showResult(data) {
    resultArea.hidden = false;
    predictionClass.textContent = data.class;

    // Parse confidence string "99.99%" -> 99.99
    const confVal = parseFloat(data.confidence);
    confidenceBar.style.width = `${confVal}%`;
    predictionConfidence.textContent = data.confidence;
}

// 4. Retraining Trigger
retrainBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to trigger model retraining? This may take a while.')) return;

    retrainBtn.disabled = true;
    trainMsg.textContent = 'Triggering training...';
    trainMsg.style.color = 'var(--text-secondary)';

    try {
        const response = await fetch(`${API_URL}/train?force=true`, {
            method: 'POST'
        });

        if (response.ok) {
            const data = await response.json();
            trainMsg.textContent = `Training started! (PID: ${data.pid})`;
            trainMsg.style.color = 'var(--success)';
        } else {
            throw new Error('Failed to start');
        }
    } catch (error) {
        trainMsg.textContent = 'Error triggering training.';
        trainMsg.style.color = 'var(--error)';
    } finally {
        setTimeout(() => {
            retrainBtn.disabled = false;
            // trainMsg.textContent = '';
        }, 5000);
    }
});
