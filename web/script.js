const API_URL = '';

// Elements
const apiStatus = document.getElementById('api-status');
const modelStatus = document.getElementById('model-status');
const metricUptime = document.getElementById('metric-uptime');
const metricPredictions = document.getElementById('metric-predictions');
const metricInference = document.getElementById('metric-inference');
const metricCpu = document.getElementById('metric-cpu');

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

// ... (other elements)

// 1. Check API Status & Metrics
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            apiStatus.textContent = 'API Online';
            apiStatus.className = 'status-badge online';
            modelStatus.textContent = data.model_status;

            // Update Metrics
            metricUptime.textContent = formatUptime(data.uptime);
            metricPredictions.textContent = data.total_predictions;
            metricInference.textContent = `${data.avg_inference.toFixed(1)}s`;
            metricCpu.textContent = `${data.cpu_usage}%`;
        } else {
            throw new Error('API Error');
        }
    } catch (error) {
        apiStatus.textContent = 'API Offline';
        apiStatus.className = 'status-badge offline';
        modelStatus.textContent = 'Unreachable';
    }
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
}

// Initial check and poll every 2s for real-time feel
checkStatus();
setInterval(checkStatus, 2000);

// 2. Image Upload Handling
dropZone.addEventListener('click', (e) => {
    console.log('Drop zone clicked');
    fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = '#475569';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    console.log('File dropped');
    dropZone.style.borderColor = '#475569';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    console.log('File input changed');
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
// ... (previous code)

// 5. Visualization Refresh - REMOVED (Replaced by System Monitoring)

// 5. Upload Training Data
const trainFileInput = document.getElementById('train-file-input');
const uploadTrainBtn = document.getElementById('upload-train-btn');
const uploadMsg = document.getElementById('upload-msg');

uploadTrainBtn.addEventListener('click', async () => {
    const file = trainFileInput.files[0];
    if (!file) {
        alert('Please select a .zip file first.');
        return;
    }

    uploadTrainBtn.disabled = true;
    uploadMsg.textContent = 'Uploading and extracting...';
    uploadMsg.style.color = 'var(--text-secondary)';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/upload_data`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            uploadMsg.textContent = 'Success! Data added.';
            uploadMsg.style.color = 'var(--success)';
            trainFileInput.value = ''; // Clear input
        } else {
            const errData = await response.json();
            throw new Error(errData.detail || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadMsg.textContent = `Error: ${error.message}`;
        uploadMsg.style.color = 'var(--error)';
    } finally {
        setTimeout(() => {
            uploadTrainBtn.disabled = false;
        }, 2000);
    }
});

// 6. Retraining Trigger
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

            // Poll for completion (simple timeout for demo, ideally use WebSocket or polling API)
            setTimeout(() => {
                trainMsg.textContent = 'Training complete!';
            }, 10000); // Refresh after 10s (approx training time)

        } else {
            throw new Error('Failed to start');
        }
    } catch (error) {
        trainMsg.textContent = 'Error triggering training.';
        trainMsg.style.color = 'var(--error)';
    } finally {
        setTimeout(() => {
            retrainBtn.disabled = false;
        }, 5000);
    }
});
