# MedGAN: AI-Powered Brain Tumor Imaging

<img src="static/css/Blue_ABstract_Brain_Technology_Logo__1_-removebg-preview.png" alt="MedGAN Logo" width="120" style="margin-bottom: 20px;">

## Overview

**MedGAN** is an end-to-end AI project designed and developed by me to generate high-quality synthetic brain tumor MRI images using advanced Generative Adversarial Networks (GANs).  
The project focuses on improving data availability and diversity for brain tumor analysis by leveraging multiple state-of-the-art GAN architectures, along with an integrated web interface for image generation and tumor detection.

This project demonstrates hands-on experience in **deep learning, medical imaging, GAN architectures, and full-stack AI deployment**.

---

## Key Features

### Multi-Architecture GAN Implementation
Implemented and trained multiple GAN variants specifically for brain tumor MRI synthesis:
- DCGAN (Deep Convolutional GAN)
- ProGAN (Progressive Growing of GANs)
- StyleGAN2 (Style-based Generator)
- WGAN-GP (Wasserstein GAN with Gradient Penalty)

### Web-Based AI Application
- Upload and analyze brain MRI images
- Generate synthetic brain tumor MRI scans
- Tumor type detection using deep learning
- Interactive Flask-based web interface

### Pre-trained Deep Learning Models
- Synthetic data generation for:
  - Glioma
  - Meningioma
  - Pituitary tumors
- Vision Transformer (ViT) based tumor classification model  
  - Achieved approximately **92% accuracy**

---

## Architecture Performance Comparison

| GAN Architecture | Image Quality | Training Stability | Diversity | Training Speed |
|------------------|--------------|-------------------|--------------|----------------|
| ProGAN           | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐            
| StyleGAN2        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐ | ⭐⭐             
| WGAN-GP          | ⭐⭐⭐      | ⭐⭐⭐⭐        | ⭐⭐⭐      | ⭐⭐⭐⭐           
| DCGAN            | ⭐⭐⭐      | ⭐⭐⭐           | ⭐⭐        | ⭐⭐⭐⭐⭐          

---

## Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Computer Vision:** OpenCV  
- **Web Framework:** Flask  
- **Models:** GANs, Vision Transformer (ViT)  
- **Dataset:** Brain Tumor MRI Dataset (Kaggle)

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 1.9+
- Flask

### Clone the Repository
```bash
git clone https://github.com/gagan-1211/Major.git
cd Major



### Installation

```bash
git clone https://github.com/gagan-1211/Major.git
cd Major
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Project Motivation

Medical imaging datasets are often limited due to privacy concerns and restricted access.  
MedGAN was built to generate high-quality synthetic brain tumor MRI images that can be used for:
- Model training
- Research experimentation
- Academic learning and prototyping

---

## Future Enhancements

- Integration of diffusion-based generative models
- Improved tumor classification accuracy
- Cloud deployment (AWS/GCP/Azure)
- Explainable AI (XAI) for model interpretability

---


## License

This project is licensed under the MIT License.
