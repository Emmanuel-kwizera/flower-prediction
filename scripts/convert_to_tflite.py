import tensorflow as tf
import pathlib
import os

# Fix for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_PATH = 'models/model.h5'
TFLITE_PATH = 'models/model.tflite'

def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations (quantization) to reduce size further
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    print(f"Saving to {TFLITE_PATH}...")
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print("Conversion complete.")

if __name__ == "__main__":
    convert()
