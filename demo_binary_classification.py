#!/usr/bin/env python3
"""
Demonstration Script for YOLO v8 Ambulance vs Non-Ambulance Binary Classification

This script demonstrates how the updated YOLO model can distinguish between:
- Class 0: non-ambulance (emergency vehicles that are not ambulances)
- Class 1: ambulance

The script shows dataset statistics and runs inference on test images.
"""

import os
import sys
import glob
from pathlib import Path
from ultralytics import YOLO

def show_dataset_info():
    """Display information about the dataset"""
    print("=" * 60)
    print("YOLO v8 AMBULANCE vs NON-AMBULANCE BINARY CLASSIFICATION")
    print("=" * 60)
    print()
    print("Dataset Information:")
    print("-------------------")
    
    # Count training images
    train_images = len(glob.glob("database/train/images/*.jpg"))
    valid_images = len(glob.glob("database/valid/images/*.jpg"))
    test_images = len(glob.glob("database/test/images/*.jpg"))
    
    print(f"Training images: {train_images}")
    print(f"Validation images: {valid_images}")
    print(f"Test images: {test_images}")
    print(f"Total images: {train_images + valid_images + test_images}")
    print()
    
    # Count labels by class
    def count_class_instances(label_dir, class_id):
        count = 0
        for label_file in glob.glob(f"{label_dir}/*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip().startswith(f"{class_id} "):
                        count += 1
        return count
    
    train_non_ambulance = count_class_instances("database/train/labels", 0)
    train_ambulance = count_class_instances("database/train/labels", 1)
    
    valid_non_ambulance = count_class_instances("database/valid/labels", 0)
    valid_ambulance = count_class_instances("database/valid/labels", 1)
    
    test_non_ambulance = count_class_instances("database/test/labels", 0)
    test_ambulance = count_class_instances("database/test/labels", 1)
    
    print("Class Distribution:")
    print("------------------")
    print(f"Class 0 (non-ambulance):")
    print(f"  Train: {train_non_ambulance}")
    print(f"  Valid: {valid_non_ambulance}")
    print(f"  Test:  {test_non_ambulance}")
    print(f"  Total: {train_non_ambulance + valid_non_ambulance + test_non_ambulance}")
    print()
    print(f"Class 1 (ambulance):")
    print(f"  Train: {train_ambulance}")
    print(f"  Valid: {valid_ambulance}")
    print(f"  Test:  {test_ambulance}")
    print(f"  Total: {train_ambulance + valid_ambulance + test_ambulance}")
    print()
    
    total_instances = (train_non_ambulance + train_ambulance + 
                      valid_non_ambulance + valid_ambulance + 
                      test_non_ambulance + test_ambulance)
    print(f"Total labeled instances: {total_instances}")
    print()

def test_model_if_available():
    """Test the model if training is complete"""
    
    # Check for trained model
    model_paths = [
        "runs/train/ambulance_detection/weights/best.pt",
        "runs/train/test_ambulance_detection/weights/best.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No trained model found. Please run training first:")
        print("  python train.py")
        return
    
    print(f"Testing with model: {model_path}")
    print("-" * 40)
    
    # Load model
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")
    print()
    
    # Test on a few specific images
    test_images = glob.glob("database/test/images/*.jpg")[:10]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Testing on {len(test_images)} sample images:")
    print("-" * 40)
    
    results = model.predict(
        source=test_images,
        save=True,
        save_txt=True,
        conf=0.25,
        project='runs/detect',
        name='demo_detection',
        exist_ok=True,
        verbose=False
    )
    
    # Analyze results
    total_detections = 0
    ambulance_detections = 0
    non_ambulance_detections = 0
    
    for i, (r, img_path) in enumerate(zip(results, test_images)):
        img_name = os.path.basename(img_path)
        print(f"{i+1:2d}. {img_name[:50]:<50}", end="")
        
        if r.boxes is not None and len(r.boxes) > 0:
            detections = []
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])
                detections.append(f"{class_name}({conf:.2f})")
                
                total_detections += 1
                if class_id == 1:
                    ambulance_detections += 1
                else:
                    non_ambulance_detections += 1
            
            print(f" -> {', '.join(detections)}")
        else:
            print(" -> No detections")
    
    print()
    print("Detection Summary:")
    print("-" * 20)
    print(f"Total detections: {total_detections}")
    print(f"Ambulance detections: {ambulance_detections}")
    print(f"Non-ambulance detections: {non_ambulance_detections}")
    print()
    print(f"Results saved to: runs/detect/demo_detection")

def main():
    """Main function"""
    # Change to repo directory
    os.chdir('/home/runner/work/Yolo_v8_emergency/Yolo_v8_emergency')
    
    # Show dataset information
    show_dataset_info()
    
    # Test model if available
    test_model_if_available()
    
    print("\nTo train the model, run:")
    print("  python train.py")
    print("\nTo run detection on all test images, run:")
    print("  python detect.py --source database/test/images")

if __name__ == "__main__":
    main()