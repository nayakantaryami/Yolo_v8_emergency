#!/usr/bin/env python3
"""
YOLO v8 Ambulance Detection Demo
A simple demo script to test the ambulance detection system.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def demo_quick_test():
    """
    Quick demo using a pre-trained YOLO model on the test dataset
    This is useful for testing before training a custom model
    """
    print("Running quick demo with pre-trained YOLOv8 model...")
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Test on a few images from the test set
    test_images_dir = "database/test/images"
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        return
    
    # Get first 5 test images
    test_images = list(Path(test_images_dir).glob("*.jpg"))[:5]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Testing on {len(test_images)} images...")
    
    # Run detection
    for img_path in test_images:
        print(f"Processing: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            save=True,
            project="runs/demo",
            name="quick_test",
            exist_ok=True,
            conf=0.25,
            verbose=False
        )
    
    print("Quick demo completed! Check 'runs/demo/quick_test' for results.")


def demo_custom_model():
    """
    Demo using the trained custom model
    """
    model_path = "runs/train/ambulance_detection/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("Custom model not found. Please train the model first using:")
        print("python train.py")
        return
    
    print("Running demo with custom trained model...")
    
    # Load custom model
    model = YOLO(model_path)
    
    # Test on the test dataset
    results = model.predict(
        source="database/test/images",
        save=True,
        save_txt=True,
        project="runs/demo",
        name="custom_model_test",
        exist_ok=True,
        conf=0.25
    )
    
    print("Custom model demo completed! Check 'runs/demo/custom_model_test' for results.")
    
    # Print some statistics
    total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"Total ambulance detections: {total_detections}")


def validate_dataset():
    """
    Validate the dataset structure and show some statistics
    """
    print("Validating dataset...")
    
    # Check dataset structure
    required_dirs = [
        "database/train/images",
        "database/train/labels",
        "database/valid/images", 
        "database/valid/labels",
        "database/test/images",
        "database/test/labels"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing: {dir_path}")
            return False
        else:
            print(f"✅ Found: {dir_path}")
    
    # Count files
    train_images = len(list(Path("database/train/images").glob("*.jpg")))
    train_labels = len(list(Path("database/train/labels").glob("*.txt")))
    valid_images = len(list(Path("database/valid/images").glob("*.jpg")))
    valid_labels = len(list(Path("database/valid/labels").glob("*.txt")))
    test_images = len(list(Path("database/test/images").glob("*.jpg")))
    test_labels = len(list(Path("database/test/labels").glob("*.txt")))
    
    print(f"\nDataset Statistics:")
    print(f"Training: {train_images} images, {train_labels} labels")
    print(f"Validation: {valid_images} images, {valid_labels} labels")
    print(f"Test: {test_images} images, {test_labels} labels")
    print(f"Total: {train_images + valid_images + test_images} images")
    
    return True


def main():
    print("🚑 YOLO v8 Ambulance Detection Demo 🚑")
    print("=" * 50)
    
    # Validate dataset first
    if not validate_dataset():
        print("❌ Dataset validation failed!")
        return
    
    print("\nChoose an option:")
    print("1. Quick test with pre-trained model")
    print("2. Test with custom trained model")
    print("3. Both")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        demo_quick_test()
    elif choice == "2":
        demo_custom_model()
    elif choice == "3":
        demo_quick_test()
        print("\n" + "="*50)
        demo_custom_model()
    else:
        print("Invalid choice!")
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("- To train a custom model: python train.py")
    print("- To run inference: python detect.py --source <image_path>")
    print("- For real-time detection: python detect.py --webcam")


if __name__ == "__main__":
    main()