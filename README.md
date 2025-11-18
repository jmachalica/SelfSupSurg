# Self-Supervised Learning for the Analysis of Robotic Surgery Recordings

This repository contains the code accompanying my MSc thesis on **self-supervised learning for surgical video understanding**.  
The work and implementation are **based on and inspired by**:

- **Dissecting Self-Supervised Learning Methods for Surgical Computer Vision**  
  Sanat Ramesh*, Vinkle Srivastav*, Deepak Alapatt, Tong Yu, Aditya Murali, Luca Sestini,  
  Chinedu Innocent Nwoye, Idris Hamoud, Saurav Sharma, Antoine Fleurentin,  
  Georgios Exarchakis, Alexandros Karargyris, Nicolas Padoy  
  ([arXiv:2207.00449](https://arxiv.org/abs/2207.00449))

- The official **SelfSupSurg** codebase released by the authors  
  (training and evaluation pipeline, configuration structure, and many implementation details).

- The **VISSL** library ([facebookresearch/vissl](https://github.com/facebookresearch/vissl)),  
  which provides the underlying framework for contrastive/self-supervised pre-training.

- The **Surgical Visual Understanding (SurgVU) Dataset**  
  ([arXiv:2501.09209](https://arxiv.org/abs/2501.09209))  
  which is the primary dataset used in this work.

## Overview

This repository is a **fork of SelfSupSurg** adapted specifically for the **SurgVU dataset**. The codebase integrates the original SelfSupSurg framework with dataset-specific modifications to enable self-supervised pre-training and fine-tuning on surgical video recordings.

### Repository Structure

- **`vissl/`** — Core modifications for dataset integration. Contains the main adaptations to work with the SurgVU dataset, including data loaders.

- **`configs/`** — Training configuration files adapted and modified for the SurgVU dataset. Includes configs for different self-supervised methods (MoCo, SimCLR, etc.) and fine-tuning protocols.

- **`slurm_scripts/`** — SLURM batch scripts used for running experiments on PLGrid. Includes scripts for data preparation, pre-training, fine-tuning, testing, and monitoring.

- **`utils/`** — General utility functions, mostly preserved from the original SelfSupSurg repository.

- **`utils/surgvu/`** — SurgVU-specific scripts and notebooks for:
  - Dataset preparation and preprocessing
  - Data cleaning and quality control
  - Exploratory Data Analysis (EDA)
  - Label processing and data splits generation

The repository adapts these components to my thesis setting (dataset, evaluation protocol and experiments),
while keeping the core ideas and training strategies from SelfSupSurg and VISSL.
---

🚧 **Detailed installation, dataset preparation and training instructions will be published soon.**  
For now, please refer to the original SelfSupSurg documentation and VISSL docs as a technical baseline.