# Image-classification-using-DL
Human vs AI-generated image classification using Deep Learning and Ensemble approach



## Overview
This repository contains an advanced deep learning framework for detecting AI-generated images. The system leverages an ensemble approach combining multiple vision models with metadata analysis to achieve high accuracy classification between AI-generated and human-created images.

## Architecture
The core of the system is the `AdvancedCLIPDetector` which employs:
- Multiple CLIP variants (ViT-L-14, ViT-H-14, ViT-B-16) with different pretraining strategies
- EfficientNetV2-L
- ConvNext Base
- Metadata feature extraction
- Attention-based feature weighting
- Multi-layer classifier with dropout and residual connections

## Features
- **Advanced Image Processing**:
  - Sophisticated image augmentation using RandAugment, ColorJitter, and RandomErasing
  - Test Time Augmentation (TTA) during inference
  - Comprehensive metadata extraction

- **Robust Training Pipeline**:
  - K-fold cross-validation (default: 5 folds)
  - Focal Loss for class imbalance handling
  - Mixed precision training
  - Gradient clipping
  - Cosine learning rate scheduling with warmup
  - Component-specific learning rates

- **State-of-the-Art Inference**:
  - Ensemble predictions across folds
  - Soft voting mechanism for final classification
  - TTA during validation and testing

## Technical Highlights
- Gradient analysis using Sobel operators
- JPEG compression artifact detection
- Statistical feature extraction
- Attention mechanisms for feature importance weighting

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-image-detection.git
cd ai-image-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
from detector import AdvancedCLIPDetector

# Initialize the detector
detector = AdvancedCLIPDetector(
    pretrained=True,
    num_folds=5
)

# Detect if an image is AI-generated
result = detector.predict("path/to/image.jpg")
probability = result['ai_probability']
is_ai = result['is_ai']  # Boolean classification

print(f"AI-generated probability: {probability:.2f}")
print(f"Classification: {'AI-generated' if is_ai else 'Human-created'}")
```

## Training Custom Models
```python
from detector import AdvancedCLIPDetector, DataLoader

# Prepare your dataset
train_loader = DataLoader("path/to/training/data", batch_size=32)
val_loader = DataLoader("path/to/validation/data", batch_size=32)

# Initialize and train the model
model = AdvancedCLIPDetector(pretrained=False)
model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Save the trained model
model.save("path/to/save/model")
```

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- timm
- open_clip_torch
- PIL
- numpy
- pandas
- scikit-learn
- albumentations

## License
[MIT License](LICENSE)

## Citation
If you use this code in your research, please cite:
```
@article{yourname2025aidetection,
  title={Advanced AI-Generated Image Detection using Ensemble Learning},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
