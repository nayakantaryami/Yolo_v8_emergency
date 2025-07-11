# 🚑 YOLO v8 Emergency Vehicle Detection

This repository contains a complete implementation of YOLO v8 for detecting ambulances and emergency vehicles using a custom dataset.

## 📋 Dataset Information

The dataset contains **492 images** in total:
- **Training**: 344 images
- **Validation**: 99 images  
- **Test**: 49 images

**Classes**:
- `0`: Emergency Vehicle
- `1`: Ambulance

The dataset is already annotated in YOLO format with normalized bounding box coordinates.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Demo

Test the system with a quick demo:

```bash
python demo.py
```

### 3. Train Custom Model

Train YOLO v8 on the ambulance dataset:

```bash
python train.py
```

### 4. Run Inference

Detect ambulances in images:

```bash
# On test images
python detect.py --source database/test/images

# On single image
python detect.py --source path/to/image.jpg

# Real-time webcam detection
python detect.py --webcam
```

## 📁 Project Structure

```
Yolo_v8_emergency/
├── database/                    # Dataset directory
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── labels/             # Training labels (YOLO format)
│   ├── valid/
│   │   ├── images/             # Validation images
│   │   └── labels/             # Validation labels
│   └── test/
│       ├── images/             # Test images
│       └── labels/             # Test labels
├── data.yaml                   # Dataset configuration
├── train.py                    # Training script
├── detect.py                   # Inference script
├── demo.py                     # Demo script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Scripts Description

### `train.py`
Trains a YOLOv8 model on the custom ambulance dataset.

**Features**:
- Uses YOLOv8 nano model for faster training
- Configurable epochs, batch size, and image size
- Early stopping with patience
- Automatic model validation
- Saves training checkpoints

**Usage**:
```bash
python train.py
```

### `detect.py`
Performs inference using the trained model.

**Features**:
- Batch processing of images
- Real-time webcam detection
- Configurable confidence threshold
- Saves results with annotations
- Outputs detection statistics

**Usage**:
```bash
# Basic usage
python detect.py

# Custom source and confidence
python detect.py --source path/to/images --conf 0.5

# Webcam detection
python detect.py --webcam

# Specify custom model
python detect.py --model path/to/model.pt
```

### `demo.py`
Interactive demo script for testing the system.

**Features**:
- Dataset validation
- Quick test with pre-trained model
- Custom model testing
- Dataset statistics

**Usage**:
```bash
python demo.py
```

## 📊 Training Configuration

The training uses the following default parameters:
- **Model**: YOLOv8 nano (`yolov8n.pt`)
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 16
- **Device**: CPU (can be changed to GPU)
- **Early Stopping**: 10 epochs patience

## 🎯 Model Performance

After training, the model will be evaluated on the validation set. Key metrics include:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95

Results are saved in `runs/train/ambulance_detection/`

## 🔍 Inference Options

### Command Line Arguments for `detect.py`:

- `--model`: Path to trained model (default: `runs/train/ambulance_detection/weights/best.pt`)
- `--source`: Source for detection (image, video, directory)
- `--conf`: Confidence threshold (default: 0.25)
- `--webcam`: Enable real-time webcam detection
- `--save-dir`: Directory to save results (default: `runs/detect`)

## 📈 Results

Detection results are saved with:
- **Annotated images**: Original images with bounding boxes and labels
- **Text files**: Detection coordinates and confidence scores
- **Statistics**: Summary of detections found

## 🛠️ Customization

### Modifying Training Parameters

Edit the `train_yolo_v8()` function in `train.py`:

```python
results = model.train(
    data='data.yaml',
    epochs=200,          # Increase epochs
    imgsz=640,
    batch=32,           # Increase batch size
    device='cuda',      # Use GPU
    # ... other parameters
)
```

### Adding New Classes

1. Update the label files with new class IDs
2. Modify `data.yaml` to include new class names:

```yaml
names:
  0: emergency_vehicle
  1: ambulance
  2: fire_truck        # New class
  3: police_car        # New class
```

## 🚨 Troubleshooting

### Common Issues:

1. **Model not found**: Make sure to train the model first using `python train.py`
2. **CUDA errors**: If you encounter GPU issues, change device to 'cpu' in training script
3. **Import errors**: Install all dependencies with `pip install -r requirements.txt`
4. **Dataset path issues**: Ensure the `database` directory structure is correct

## 📄 License

This project uses the dataset provided under CC BY 4.0 license via Roboflow.

## 🤝 Contributing

Feel free to contribute by:
- Adding more training data
- Improving model architecture
- Optimizing inference speed
- Adding new features

## 📞 Support

For issues and questions, please create an issue in the repository.