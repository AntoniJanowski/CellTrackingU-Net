# CNN-based Segmentation of Cells in Multi-modal Microscopy Images

This project was created by me and three other colleagues for the **Deep Learning in Life Sciences Hackathon 2024** in Heidelberg. It was the final part of the *Deep Learning in Life Sciences* course at the Faculty of Mathematics, University of Warsaw. The best-performing students from the course were selected to participate in the hackathon alongside students from many other European universities.

## Overview
Cell segmentation is one of the central tasks in biomedical image analysis. It enables counting the number of cells, quantifying single-cell fluorescence intensity, and tracking cells to analyze their motion.

This project aims to develop a Convolutional Neural Network (CNN) for cell segmentation in microscopy images.

![Example U-Net output](images/unet_output.png)

## Model and Data
The model used is a 2D U-Net with zero-padding in the convolution layers.  
The datasets comprise 2D time-lapse microscopy images from the **Cell Tracking Challenge**, covering three different imaging modalities:

- Fluorescence microscopy: HeLa cells stably expressing H2B-GFP
- Differential interference contrast (DIC) microscopy: HeLa cells on a flat glass
- Phase-contrast microscopy: Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate

The datasets were taken from the official challenge page: https://celltrackingchallenge.net/2d-datasets/.

## My Contributions
- Implemented data augmentation using NumPy
- Wrote the PyTorch code for the U-Net module, including GPU-powered training, validation, and model saving
- Enabled training on single or multiple imaging modalities
- Implemented metrics such as pixel-wise accuracy and true positive rate, and visualizations for model outputs
- Trained multiple models and conducted hyperparameter search

## Technologies Used
- Python
- PyTorch
- PyTorch Lightning
- NumPy

## Code Structure
- `U_Net.py`: Contains the U-Net model implementation  
- `notebooks/Training_pipeline.ipynb`: Jupyter notebook showing data loading, augmentation, and training steps  
- `Train_U_Net_example.py`: Script for training the model; set the path to the preprocessed data folder inside the script before running
Additional experimental notebooks can be found in notebooks/_WIP/.
