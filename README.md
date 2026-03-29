**[中文版](README_CN.md) | English**

# Image-Based Root Canal Treatment Evaluation System

This project is a deep learning-based automatic evaluation system for root canal treatment outcomes. The system performs image segmentation and classification on periapical radiographs to automatically assess the quality of root canal filling, outputting evaluation results in four grades (A/B/C/D). A visual GUI built with PyQt5 supports one-click evaluation and model training.

## System Architecture

The system adopts a three-stage pipeline architecture of **"Segmentation + Feature Fusion + Classification"**:

```
Periapical Radiograph Input
           |
           v
    +--------------+
    |    UNet++     |  -- Semantic Segmentation: Extract tooth region mask + root canal region mask
    +--------------+
           |
           v
    +--------------+
    | Feature Fusion|  -- Superposition: Combine original image(R), tooth mask(G), root canal mask(B) into RGB
    +--------------+
           |
           v
    +--------------+
    |   ResNet50    |  -- Image Classification: Grade root canal filling quality from fused image
    +--------------+
           |
           v
    Evaluation Result (A/B/C/D)
```

### GUI

The system provides three operation interfaces:

- **Main Interface**: Navigation entry for jumping to the evaluation or training interface
- **Evaluation Interface**: Upload periapical radiographs, select models, and complete the full segmentation-fusion-classification pipeline with one click
- **Training Interface**: Select dataset and output paths, configure training parameters (epochs, learning rate, batch size), and train UNet++ or ResNet models

## Project Structure

```
├── main.py                 # Application entry point
├── requirements.txt        # Dependency list
├── class_indices.json      # Classification label mapping {"0":"A", "1":"B", "2":"C", "3":"D"}
│
├── archs.py                # UNet++ (NestedUNet) network architecture
├── model.py                # ResNet network architecture (supports ResNet18/34/50/101/152)
├── dataset.py              # Custom Dataset for loading images and masks
├── losses.py               # BCEDiceLoss loss function
├── metrics.py              # IoU and Dice coefficient computation
├── utils.py                # Utility functions (AverageMeter, etc.)
│
├── train_u.py              # UNet++ segmentation model training (QThread)
├── train_c.py              # ResNet classification model training (QThread)
├── val.py                  # Segmentation model inference/validation
├── predict.py              # Classification model inference
├── trans_3gto1.py          # Superposition feature fusion (original + tooth mask + root canal mask -> RGB)
│
├── page_main.py            # Main interface UI layout
├── page_maincode.py        # Main interface logic
├── page1.py                # Evaluation interface UI layout
├── page1_code.py           # Evaluation interface logic
├── page2.py                # Training interface UI layout
├── page2_code.py           # Training interface logic
│
├── inputs/                 # Segmentation dataset directory
│   └── root_dataset/
│       ├── images/         # Input images (1:1 aspect ratio, padded with black borders)
│       └── masks/
│           ├── 0/          # Tooth region masks
│           └── 1/          # Root canal region masks
│
├── models/                 # Trained model weights
│   ├── resNet50.pth        # ResNet50 classification model weights
│   └── root_dataset_NestedUNet_woDS/
│       ├── model.pth       # UNet++ segmentation model weights
│       ├── config.yml      # Training configuration
│       └── log.csv         # Training log
│
├── outputs/                # Segmentation output results
│   └── root_dataset_NestedUNet_woDS/
│       ├── 0/              # Tooth region segmentation results
│       └── 1/              # Root canal region segmentation results
│
└── rgb/                    # Superposition feature fusion output images
```

## Requirements and Installation

### Requirements

- **Python 3.9** (required; PyTorch may have compatibility issues with higher versions)
- CUDA (optional, for GPU-accelerated training)

### Installation Steps

1. **Create a virtual environment**

```bash
python -m venv venv
```

2. **Activate the virtual environment**

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note**: `albumentations` must be below version 2.0 (`albumentations<2.0`). Version 2.0+ has breaking API changes (the `albumentations.augmentations.transforms` module was removed).

### Known Issues

- **GUI scaling not adaptive**: The GUI was designed at 100% display scaling on Windows. If text appears truncated or missing, try adjusting the system display scaling factor. On macOS, font display issues may also occur due to scaling differences.

## Usage

### Launch the Application

```bash
python main.py
```

### Evaluation Workflow

1. Click to enter the **Evaluation Interface** from the main page
2. Upload the periapical radiograph to be evaluated
3. Select the segmentation model (UNet++) and classification model (ResNet50) weight files
4. Click "Start Evaluation". The system will sequentially perform:
   - UNet++ segmentation -> Extract tooth and root canal region masks
   - Superposition feature fusion -> Generate RGB three-channel fused image
   - ResNet50 classification -> Output filling quality grade
5. View the evaluation result (grade + confidence + Top-4 prediction probabilities)

### Training Workflow

1. Click to enter the **Training Interface** from the main page
2. Select the model type to train (UNet++ segmentation / ResNet classification)
3. Set the dataset path and model output path
4. Configure training parameters (epochs, learning rate, batch size)
5. Click "Start Training" and monitor progress on the interface

## Model Details

### UNet++ Segmentation Model

- **Architecture**: NestedUNet (UNet++ nested U-shaped network)
- **Basic Block**: VGGBlock (Conv-BN-ReLU x 2)
- **Channel Configuration**: [32, 64, 128, 256, 512]
- **Input Size**: 192 x 192 x 3
- **Output Classes**: 2 (tooth region + root canal region)
- **Loss Function**: BCEDiceLoss (BCE + Dice combined loss)
- **Optimizer**: SGD (lr=1e-3, momentum=0.9, weight_decay=1e-4)
- **LR Scheduler**: CosineAnnealingLR
- **Data Augmentation**: Random 90-degree rotation, flipping, random hue/saturation/brightness/contrast adjustment

### ResNet50 Classification Model

- **Architecture**: ResNet50 (50-layer residual network)
- **Input Size**: 224 x 224 x 3 (RGB fused image)
- **Output Classes**: 4 (A / B / C / D)
- **Image Normalization**: mean=[0.237, 0.207, 0.044], std=[0.242, 0.374, 0.160]

### Evaluation Grade Definitions

| Grade | Description |
|-------|-------------|
| **A** | Adequate root canal filling with uniform density and good quality |
| **B** | Slightly underfilled or overfilled, but within acceptable range |
| **C** | Significantly insufficient or excessive filling, requires attention |
| **D** | Severely inadequate filling, retreatment recommended |

When the Top-1 prediction confidence is below 0.7, the system outputs a fuzzy grade (e.g., A~B, B~C), indicating the result falls between two grades.

### Feature Fusion Method -- Superposition

The segmentation results and original image are fused into a single RGB image for the classification network:

- **R Channel**: Original periapical radiograph grayscale image (192 x 192)
- **G Channel**: Tooth region segmentation mask (192 x 192)
- **B Channel**: Root canal region segmentation mask (192 x 192)

This method enables the classification network to simultaneously utilize original image texture information and spatial location information of segmented regions.

## Experimental Results

The following results are from the thesis experiments (dataset contains 500+ periapical radiographs):

### Segmentation Performance

| Metric | Value |
|--------|-------|
| Mean IoU | 0.7634 |

### Classification Performance (Superposition + ResNet50)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 59.40% |
| Top-2 Accuracy | 86.22% |

## References and Acknowledgments

### Original Code Repositories

- **UNet++ (pytorch-nested-unet)**: [https://github.com/4uiiurz1/pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet)

  The UNet++ segmentation network architecture in this project is based on this repository.

- **ResNet (PyTorch official torchvision)**: [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

  The ResNet classification network in this project references the ResNet implementation in torchvision.

### Reference Papers

- Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). **UNet++: A Nested U-Net Architecture for Medical Image Segmentation.** *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, pp. 3-11.

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778.

### Related Technologies

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [albumentations](https://github.com/albumentations-team/albumentations) - Image augmentation library
- [OpenCV](https://opencv.org/) - Image processing library
