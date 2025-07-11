#!/usr/bin/env python3
"""
Dataset Visualization Utility
Visualizes the ambulance dataset with bounding box annotations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def read_yolo_label(label_path, img_width, img_height):
    """
    Read YOLO format label file and convert to pixel coordinates
    
    Args:
        label_path (str): Path to label file
        img_width (int): Image width
        img_height (int): Image height
    
    Returns:
        list: List of bounding boxes [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                boxes.append((class_id, x1, y1, x2, y2))
    
    return boxes


def visualize_sample_images(dataset_split="train", num_samples=6):
    """
    Visualize sample images with annotations
    
    Args:
        dataset_split (str): Dataset split ('train', 'valid', 'test')
        num_samples (int): Number of samples to visualize
    """
    class_names = {0: "Emergency Vehicle", 1: "Ambulance"}
    colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Blue for emergency, Green for ambulance
    
    images_dir = f"database/{dataset_split}/images"
    labels_dir = f"database/{dataset_split}/labels"
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_files = list(Path(images_dir).glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    # Randomly sample images
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create subplot grid
    cols = 3
    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, img_path in enumerate(sample_images):
        row = idx // cols
        col = idx % cols
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # Read corresponding label
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        boxes = read_yolo_label(str(label_path), img_width, img_height)
        
        # Draw bounding boxes
        for class_id, x1, y1, x2, y2 in boxes:
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = class_names.get(class_id, f"Class {class_id}")
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
        
        # Display image
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"{img_path.name}\nBoxes: {len(boxes)}")
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for idx in range(len(sample_images), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'dataset_visualization_{dataset_split}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as: dataset_visualization_{dataset_split}.png")


def analyze_dataset():
    """
    Analyze the dataset and print statistics
    """
    print("📊 Dataset Analysis")
    print("=" * 50)
    
    class_names = {0: "Emergency Vehicle", 1: "Ambulance"}
    total_stats = {0: 0, 1: 0}
    
    for split in ["train", "valid", "test"]:
        labels_dir = f"database/{split}/labels"
        
        if not os.path.exists(labels_dir):
            continue
        
        split_stats = {0: 0, 1: 0}
        total_images = 0
        total_boxes = 0
        
        label_files = list(Path(labels_dir).glob("*.txt"))
        total_images = len(label_files)
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        split_stats[class_id] += 1
                        total_stats[class_id] += 1
                        total_boxes += 1
        
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {total_images}")
        print(f"  Total boxes: {total_boxes}")
        for class_id, count in split_stats.items():
            class_name = class_names.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count}")
    
    print(f"\nOVERALL STATISTICS:")
    total_boxes = sum(total_stats.values())
    print(f"  Total boxes: {total_boxes}")
    for class_id, count in total_stats.items():
        class_name = class_names.get(class_id, f"Class {class_id}")
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")


def main():
    print("🔍 Dataset Visualization and Analysis Tool")
    print("=" * 50)
    
    # Analyze dataset
    analyze_dataset()
    
    print("\nVisualization Options:")
    print("1. Visualize training samples")
    print("2. Visualize validation samples") 
    print("3. Visualize test samples")
    print("4. All datasets")
    
    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    
    if choice == "1":
        visualize_sample_images("train")
    elif choice == "2":
        visualize_sample_images("valid")
    elif choice == "3":
        visualize_sample_images("test")
    elif choice == "4":
        for split in ["train", "valid", "test"]:
            print(f"\nVisualizing {split} set...")
            visualize_sample_images(split)
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()