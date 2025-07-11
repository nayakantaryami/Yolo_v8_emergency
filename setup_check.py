#!/usr/bin/env python3
"""
Complete YOLO v8 Ambulance Detection Workflow
This script demonstrates the complete pipeline from setup to inference.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'ultralytics',
        'cv2',
        'torch',
        'torchvision',
        'PIL',
        'numpy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True


def validate_project_structure():
    """Validate the project structure"""
    print("\n📁 Validating project structure...")
    
    required_files = [
        'data.yaml',
        'train.py',
        'detect.py',
        'demo.py',
        'visualize_dataset.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'database/train/images',
        'database/train/labels',
        'database/valid/images',
        'database/valid/labels',
        'database/test/images',
        'database/test/labels'
    ]
    
    all_valid = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_valid = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            count = len(list(Path(dir_path).glob("*")))
            print(f"✅ {dir_path} ({count} files)")
        else:
            print(f"❌ {dir_path}")
            all_valid = False
    
    return all_valid


def show_workflow():
    """Show the complete workflow"""
    print("\n🚀 Complete YOLO v8 Ambulance Detection Workflow")
    print("=" * 60)
    
    workflow_steps = [
        ("1. Setup Environment", "pip install -r requirements.txt"),
        ("2. Analyze Dataset", "python visualize_dataset.py"),
        ("3. Quick Demo", "python demo.py"),
        ("4. Train Custom Model", "python train.py"),
        ("5. Run Inference", "python detect.py --source database/test/images"),
        ("6. Real-time Detection", "python detect.py --webcam"),
        ("7. Custom Inference", "python detect.py --source <your_image.jpg>")
    ]
    
    for step, command in workflow_steps:
        print(f"\n{step}:")
        print(f"   {command}")
    
    print("\n📊 Model Training Details:")
    print("   - Model: YOLOv8 nano (fast training)")
    print("   - Dataset: 344 train, 99 valid, 49 test images")
    print("   - Classes: Emergency Vehicle (0), Ambulance (1)")
    print("   - Training time: ~15-30 minutes on CPU")
    print("   - Output: Best model saved to runs/train/ambulance_detection/weights/best.pt")
    
    print("\n🎯 Inference Options:")
    print("   - Single image: python detect.py --source image.jpg")
    print("   - Batch images: python detect.py --source image_folder/")
    print("   - Video: python detect.py --source video.mp4")
    print("   - Webcam: python detect.py --webcam")
    print("   - Custom model: python detect.py --model custom_model.pt")


def show_dataset_info():
    """Show dataset information"""
    print("\n📊 Dataset Information")
    print("=" * 40)
    
    try:
        from visualize_dataset import analyze_dataset
        analyze_dataset()
    except Exception as e:
        print(f"Could not analyze dataset: {e}")
        
        # Manual count
        for split in ['train', 'valid', 'test']:
            img_dir = f"database/{split}/images"
            if os.path.exists(img_dir):
                count = len(list(Path(img_dir).glob("*.jpg")))
                print(f"{split.upper()}: {count} images")


def main():
    print("🚑 YOLO v8 Ambulance Detection - Complete Setup & Workflow")
    print("=" * 70)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Validate structure
    structure_ok = validate_project_structure()
    
    # Show dataset info
    show_dataset_info()
    
    # Show workflow
    show_workflow()
    
    # Final status
    print("\n" + "=" * 70)
    if deps_ok and structure_ok:
        print("🎉 Project setup is complete and ready to use!")
        print("\n🚀 Quick start:")
        print("   python demo.py")
    else:
        print("⚠️  Project setup incomplete. Please fix the issues above.")
        if not deps_ok:
            print("   → Install dependencies: pip install -r requirements.txt")
        if not structure_ok:
            print("   → Check project structure and dataset")
    
    print("\n📖 For detailed instructions, see README.md")


if __name__ == "__main__":
    main()