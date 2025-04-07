# Image-Ware

A deep learning-based malware classification system that converts malware binaries into grayscale images for analysis and classification.

## Overview

Image-Ware is a PyTorch-based project that transforms malware binaries into grayscale images and uses a Convolutional Neural Network (CNN) to classify them into different malware families. The system processes malware samples, converts them to a visual representation, and trains a model to identify malware types based on these visual patterns.

## Project Structure

- `transform.py`: Converts malware binaries to grayscale images
- `compress.py`: Compresses the generated images into .npz files
- `decompress.py`: Extracts images from .npz files
- `train.py`: Trains the CNN model on the processed images
- `test.py`: Evaluates the model's performance on test data
- `data_loader.py`: Handles data loading and preprocessing
- `model.py`: Defines the CNN architecture

## Features

- **Binary to Image Conversion**: Transforms malware binaries into grayscale images
- **Adaptive Image Sizing**: Automatically adjusts image dimensions based on file size
- **Efficient Data Storage**: Compresses images into .npz files to save space
- **Deep Learning Classification**: Uses a CNN to classify malware families
- **Performance Tracking**: Tracks and reports per-class accuracy metrics

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/image-ware.git
cd image-ware
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place malware binaries in the `data/binaries` directory
2. Run the transformation script to convert binaries to images:
```
python transform.py
```

3. (Optional) Compress the generated images:
```
python compress.py
```

4. (Optional) Decompress images if needed:
```
python decompress.py
```

### Training

Train the model on the processed images:
```
python train.py
```

The training script will:
- Detect available CUDA devices
- Load the dataset
- Initialize the CNN model
- Train for the specified number of epochs
- Save checkpoints periodically

### Evaluation

Evaluate the model's performance:
```
python test.py
```

The test script will:
- Load the trained model
- Run inference on the test dataset
- Calculate and display overall accuracy
- Show per-class accuracy metrics

## Model Architecture

The project uses a Convolutional Neural Network (CNN) with the following structure:
- Multiple convolutional layers with ReLU activation
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Cross-entropy loss function
- Adam optimizer

## Data Organization

- `data/binaries/`: Contains the original malware binaries
- `data/images/`: Stores the processed grayscale images
- `npz_data/`: Contains compressed .npz files
- `sample_images/`: Stores sample PNG images for visualization
- `checkpoints/`: Stores model checkpoints during training

## License

[Specify your license here]

## Acknowledgements

- [List any acknowledgements or references]

