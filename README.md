## Exploring Advanced MRI Airway Segmentation Techniques for Voice Analysis: A Comparative Study of 3DUNet, 2DUNet, 3D Transfer Learning UNet, and 3D Transformer UNet 

## Overview

This repository contains code and resources for segmenting the human upper airway using deep learning techniques, specifically tailored for 3D MRI data. The main objective is to facilitate accurate segmentation of the vocal tract to analyze morphological changes during phonation and other vocal activities.

## Contents

- `.gitignore`: Configuration file to specify files and directories to be ignored by Git.
- `Unet2D.ipynb`: Notebook implementing 2D U-Net for segmentation tasks.
- `Unet3D_MRI.ipynb`: Notebook for 3D U-Net segmentation on MRI data.
- `Unet3D_Trans_MRI.ipynb`: Notebook for transfer learning with 3D U-Net on MRI data.
- `Unet3D_xray.ipynb`: Notebook implementing 3D U-Net for X-ray data.
- `network analysis.ipynb`: Analysis of the network performance and results.
- `requirements.txt`: List of dependencies required to run the notebooks.

## Summary

The study focuses on the human vocal tract, which includes key articulators such as the tongue, soft palate, epiglottis, and vocal folds. The research addresses the segmentation of the vocal tract using MRI, which is non-invasive and poses minimal health risks but is slower compared to other imaging modalities. Recent advancements in accelerated MRI techniques have allowed rapid imaging of the vocal tract, aiding in quantitative assessments of vocal tract posture modulation.

Deep learning, particularly U-NET models, has shown promise in segmenting the upper airway from 3D MRI volumes. However, these approaches require substantial annotated training data, often necessitating manual annotations, which is a labor-intensive process. Despite progress in semi-automatic and fully automatic segmentation methods, challenges remain, such as the scarcity of open-source datasets and the need to fully leverage 3D features to avoid non-anatomical segmentations.

## Dataset

The dataset can be downloaded from the following link: [Figshare Dataset](https://figshare.com/s/cb050b61c0189605feda).

This dataset comprises training and testing data for creating a deep learning segmentation model for the upper airway. It includes upper airway scans and their corresponding annotations, derived from the French speaker dataset. These annotations were meticulously crafted by experts for this specific project.

The data is structured as follows:

- **Training:**
  - `RTrainVolumes`: Contains 3D upper airway MRI volumes.
  - `RtrainLabels*`: Contains the corresponding segmentations.
- **Testing:**
  - `RVolumes`: Contains 3D upper airway MRI volumes.
  - `RLabels`: Contains the corresponding segmentations.

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
```
## Citation

Please cite the following papers when using the code:

1) S Erattakulangara, Exploring Advanced MRI Airway Segmentation Techniques for Voice Analysis: A Comparative Study of 3DUNet, 2DUNet, 3D Transfer Learning UNet, and 3D Transformer UNet

