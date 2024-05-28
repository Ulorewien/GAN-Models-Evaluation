# GAN-Models-Evaluation

Comparison and evaluation of Simple Conditional GAN and Style GAN architectures for conditional image generation tasks, focusing on image realism, diversity, and style fidelity using custom datasets.

## Models

1. Conditional - GAN
2. StyleGAN

## Datasets

1. [CelebA](https://pytorch.org/vision/0.17/generated/torchvision.datasets.CelebA.html)
2. [LSUN](https://pytorch.org/vision/0.16/generated/torchvision.datasets.LSUN.html)
3. [AFHQ](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## Training Specs

- Image Size: 64x64
- Learning Rate: 1e-3
- Optimizer: Adam
- Epochs: 10
- Batch Size: 16
- Seed: 123
