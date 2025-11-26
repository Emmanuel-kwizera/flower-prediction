import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import argparse
import sys

# Configuration
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATA_DIR = pathlib.Path('../data/flowers')
MODEL_PATH = '../models/model.h5'
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 15

def download_data():
    """Downloads the flower dataset if it doesn't exist."""
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found. Downloading data...")
        dataset_url = DATA_URL
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        data_dir = pathlib.Path(data_dir)
        
        # Move data to expected location if needed, or just symlink/copy
        # For simplicity in this script, we might just use the cache dir or move it.
        # However, the notebook expects '../data/flowers'. 
        # Let's ensure the directory structure matches what the user expects.
        
        # Actually, tf.keras.utils.get_file downloads to ~/.keras/datasets/ by default.
        # We want it in ../data/flowers.
        
        import shutil
        import tarfile
        import urllib.request
        
        # Create data directory if it doesn't exist
        pathlib.Path('../data').mkdir(parents=True, exist_ok=True)
        
        tgz_path = '../data/flower_photos.tgz'
        if not os.path.exists(tgz_path):
             print(f"Downloading {DATA_URL} to {tgz_path}...")
             urllib.request.urlretrieve(DATA_URL, tgz_path)
        
        print(f"Extracting {tgz_path}...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(path='../data')
        
        # Rename if necessary (tar extracts to 'flower_photos')
        if os.path.exists('../data/flower_photos') and not DATA_DIR.exists():
            os.rename('../data/flower_photos', str(DATA_DIR))
            
        print("Data acquisition complete.")
    else:
        print(f"Data directory {DATA_DIR} already exists. Skipping download.")

def check_for_retraining_need(force=False):
    """
    Checks if retraining is needed.
    This is a placeholder for more complex logic (e.g., checking for new data files,
    model drift, or a specific trigger file).
    """
    if force:
        return True
    
    # Example trigger: Check if a 'retrain.flag' file exists in data directory
    trigger_file = pathlib.Path('../data/retrain.flag')
    if trigger_file.exists():
        print("Trigger file found. Initiating retraining...")
        # Optionally remove the flag
        # trigger_file.unlink() 
        return True
    
    # If model doesn't exist, we must train
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Initiating training...")
        return True

    print("No need to retrain. Use --force to override.")
    return False

def train_model():
    """Runs the full training pipeline."""
    print("Starting training pipeline...")
    
    # 1. Data Acquisition
    download_data()
    
    # 2. Data Loading & Processing
    print("Loading data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
      DATA_DIR,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      DATA_DIR,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(IMG_HEIGHT, IMG_WIDTH),
      batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Model Creation
    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(IMG_HEIGHT,
                                      IMG_WIDTH,
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()

    # 4. Training
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS
    )

    # 5. Save Model
    # Ensure models directory exists
    pathlib.Path('../models').mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {MODEL_PATH}...")
    # Using .keras format is recommended over .h5 for newer TF versions, 
    # but sticking to .h5 to match notebook unless requested otherwise.
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = str(pathlib.Path(MODEL_PATH).with_suffix('.tflite'))
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flower Prediction Model")
    parser.add_argument('--force', action='store_true', help="Force retraining even if not triggered")
    args = parser.parse_args()

    if check_for_retraining_need(args.force):
        train_model()
