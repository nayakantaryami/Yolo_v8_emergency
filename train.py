#!/usr/bin/env python3
"""
YOLO v8 Training Script for Ambulance Detection
This script trains a YOLOv8 model on the custom ambulance dataset.
"""

import os
from ultralytics import YOLO

def train_yolo_v8():
    """
    Train YOLOv8 model on the ambulance dataset
    """
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model for faster training
    
    # Train the model
    results = model.train(
        data='data.yaml',          # path to dataset YAML
        epochs=100,                # number of training epochs
        imgsz=640,                 # training image size
        batch=16,                  # batch size
        device='cpu',              # training device ('cpu' or 'cuda')
        workers=4,                 # number of workers
        patience=10,               # early stopping patience
        save=True,                 # save training checkpoints
        save_period=10,            # save checkpoint every 10 epochs
        project='runs/train',      # project directory
        name='ambulance_detection' # experiment name
    )
    
    # Evaluate the model
    metrics = model.val()
    
    print(f"Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
    return model, results, metrics

if __name__ == "__main__":
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("Error: data.yaml not found. Please ensure the data configuration file exists.")
        exit(1)
    
    # Check if dataset directory exists
    if not os.path.exists('database'):
        print("Error: database directory not found. Please ensure the dataset is available.")
        exit(1)
    
    print("Starting YOLOv8 training for ambulance detection...")
    model, results, metrics = train_yolo_v8()
    print("Training finished successfully!")