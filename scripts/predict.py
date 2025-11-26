import os
# Fix for macOS TensorFlow/OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pathlib
import argparse
import json
import sys
import random

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("DEBUG: TensorFlow not found. Using mock prediction.", file=sys.stderr)

def predict(image_path, model_path):
    print(f"DEBUG: Starting prediction function", file=sys.stderr)
    
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    if not TF_AVAILABLE:
        # Mock prediction
        print("DEBUG: Running mock prediction", file=sys.stderr)
        predicted_class = random.choice(class_names)
        confidence = random.uniform(70.0, 99.9)
        return {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "note": "Mock prediction (TensorFlow not installed)"
        }

    from PIL import Image
    
    # Check if we have a TFLite model
    is_tflite = model_path.endswith('.tflite')
    print(f"DEBUG: Received model_path: {model_path} (TFLite: {is_tflite})", file=sys.stderr)
    
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    
    try:
        img_height = 180
        img_width = 180
        
        print(f"DEBUG: Loading image {image_path}", file=sys.stderr)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((img_height, img_width))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        print("DEBUG: Running prediction", file=sys.stderr)
        
        if is_tflite:
            # TFLite Inference
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            predictions = interpreter.get_tensor(output_details[0]['index'])
        else:
            # Fallback to Keras (legacy)
            # Disable GPU to avoid Metal/threading crashes
            try:
                tf.config.set_visible_devices([], 'GPU')
            except:
                pass
            model = tf.keras.models.load_model(model_path)
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
