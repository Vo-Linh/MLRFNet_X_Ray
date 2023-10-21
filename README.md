# MLRFNet_X_Ray

This repository contains the code for MLRFNet, a deep learning model for X-ray image analysis. MLRFNet is a multi-level residual feature network that uses a combination of residual blocks and feature fusion to extract deep features from X-ray images. MLRFNet can be used for a variety of X-ray image analysis tasks, such as classification, detection, and segmentation.

## Requirements

* Python 3.6+
* PyTorch 1.7+
* Torchvision 0.9+
* NumPy 1.19+
* tqdm 4.60+
* scikit-learn 1.0+

## Installation

To install the required dependencies, run the following command:

pip install -r requirements.txt

## Usage

To train MLRFNet, run the following command:
python main.py --config configs/mlrfnet.yaml --gpu_ids 1 --name name_exp