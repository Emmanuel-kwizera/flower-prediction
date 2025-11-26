# Flower Prediction System

A robust, containerized application for classifying flower species (Daisy, Dandelion, Rose, Sunflower, Tulip). Built with TensorFlow, FastAPI, and a modern HTML/CSS/JS frontend.

## Features
*   **Real-time Prediction:** Drag & drop interface for instant classification.
*   **System Monitoring:** Dashboard tracking uptime, total predictions, inference speed, and CPU usage.
*   **Memory Optimized:** Uses TensorFlow Lite for efficient inference (fits within 512MB RAM).
*   **Retraining Pipeline:** Trigger model retraining directly from the UI.
*   **Containerized:** Docker support for easy deployment (e.g., Render).

## Tech Stack
*   **Backend:** Python, FastAPI, TensorFlow/Keras (TFLite)
*   **Frontend:** HTML5, CSS3 , Vanilla JavaScript
*   **Deployment:** Docker, Render

## Setup & Installation

### Prerequisites
*   Python 3.9+
*   Docker (optional, for containerization)

### Local Development
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Flower-prediction
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the API:**
    ```bash
    python3 api/main.py
    ```
    The app will be available at `http://127.0.0.1:8000`.

## Docker Deployment
1.  **Build the image:**
    ```bash
    docker build -t flower-app .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8000:8000 flower-app
    ```

## API Endpoints
*   `GET /`: Redirects to Web UI.
*   `GET /health`: System status and metrics.
*   `POST /predict`: Upload image for classification.
*   `POST /train`: Trigger model retraining.

## Deployment Links
*   **Frontend:** [https://flower-prediction-app-3c3y.onrender.com](https://flower-prediction-app-3c3y.onrender.com)
*   **Backend API Docs:** [https://flower-prediction-app-3c3y.onrender.com/docs#/](https://flower-prediction-app-3c3y.onrender.com/docs#/)


## Project Structure
*   `api/`: FastAPI backend code.
*   `web/`: Frontend static files (HTML/CSS/JS).
*   `scripts/`: Training and prediction scripts.
*   `models/`: Trained model files (.h5, .tflite).
*   `visualizations/`: Generated training plots.
