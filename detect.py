#!/usr/bin/env python3
"""
YOLO v8 Inference Script for Ambulance vs Non-Ambulance Detection
This script performs inference using a trained YOLOv8 model to detect ambulances vs non-ambulances in images.
Binary classification:
- Class 0: non-ambulance (emergency vehicles that are not ambulances)
- Class 1: ambulance
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def detect_ambulances(model_path, source, save_dir="runs/detect", conf_threshold=0.25):
    """
    Detect ambulances vs non-ambulances in images or video using trained YOLOv8 model
    
    Args:
        model_path (str): Path to the trained model
        source (str): Path to image, video, or directory
        save_dir (str): Directory to save results
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        save=True,
        save_txt=True,          # Save results as txt files
        save_conf=True,         # Save confidence scores
        conf=conf_threshold,    # Confidence threshold
        project=save_dir,
        name='ambulance_detection',
        exist_ok=True
    )
    
    print(f"Detection completed! Results saved to: {save_dir}/ambulance_detection")
    
    # Print detection summary
    detection_count = 0
    for r in results:
        if r.boxes is not None:
            detection_count += len(r.boxes)
    
    print(f"Total detections: {detection_count}")
    
    return results


def detect_from_webcam(model_path, conf_threshold=0.25):
    """
    Real-time ambulance vs non-ambulance detection from webcam
    
    Args:
        model_path (str): Path to the trained model
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting real-time detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('Ambulance Detection', annotated_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Ambulance vs Non-Ambulance Detection Inference')
    parser.add_argument('--model', type=str, default='runs/train/ambulance_detection/weights/best.pt',
                        help='Path to trained model')
    parser.add_argument('--source', type=str, default='database/test/images',
                        help='Path to source (image, video, directory)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for real-time detection')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first using train.py")
        exit(1)
    
    if args.webcam:
        detect_from_webcam(args.model, args.conf)
    else:
        if not os.path.exists(args.source):
            print(f"Error: Source not found at {args.source}")
            exit(1)
        
        detect_ambulances(args.model, args.source, args.save_dir, args.conf)


if __name__ == "__main__":
    main()