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

The repository adapts these components to my thesis setting (dataset, evaluation protocol and experiments),
while keeping the core ideas and training strategies from SelfSupSurg and VISSL.
---

🚧 **Detailed installation, dataset preparation and training instructions will be published soon.**  
For now, please refer to the original SelfSupSurg documentation and VISSL docs as a technical baseline.