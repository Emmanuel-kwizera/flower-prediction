import os
# Fix for macOS TensorFlow/OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pathlib
import argparse
import json
import sys

def predict(image_path, model_path):
    from PIL import Image
    print(f"DEBUG: Starting prediction function", file=sys.stderr)
    print(f"DEBUG: Received model_path: {model_path}", file=sys.stderr)
    print(f"DEBUG: CWD: {os.getcwd()}", file=sys.stderr)
    print(f"DEBUG: Exists: {os.path.exists(model_path)}", file=sys.stderr)
    
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    
    try:
        # Disable GPU to avoid Metal/threading crashes
        try:
            print("DEBUG: Disabling GPU", file=sys.stderr)
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass

        print(f"DEBUG: Loading model from {model_path}", file=sys.stderr)
        model = tf.keras.models.load_model(model_path)
        print("DEBUG: Model loaded", file=sys.stderr)
        
        img_height = 180
        img_width = 180
        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        
        print(f"DEBUG: Loading image {image_path}", file=sys.stderr)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((img_height, img_width))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        print("DEBUG: Running prediction", file=sys.stderr)
        predictions = model.predict(img_array, verbose=0)
        print("DEBUG: Prediction complete", file=sys.stderr)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        print(f"DEBUG: Exception: {e}", file=sys.stderr)
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--model_path", default="models/model.h5", help="Path to model file")
    args = parser.parse_args()
    
    result = predict(args.image_path, args.model_path)
    print(json.dumps(result))
