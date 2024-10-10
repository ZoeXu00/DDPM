# Denoising Diffusion Probabilistic Models (DDPM) for Image Generation

## Overview
The project focuses on building a Denoising Diffusion Probabilistic Model (DDPM) for image generation, using a subset of the Animal Faces-HQ (AFHQ) dataset. It aims to enhance the generative model by implementing a cosine-based variance schedule and utilizing a U-Net architecture for the denoising process. These components are central to modern generative models, enabling the model to generate high-quality, realistic images by learning to reverse the diffusion process from noise to image.

## Dataset

The project uses a subset of the **Animal Faces-HQ (AFHQ)** dataset, which contains high-resolution images of animals. For this implementation, we focus specifically on the **cat images** subset to reduce computational complexity. The AFHQ dataset offers a rich set of images with a resolution of 512x512 pixels, ensuring detailed features that are ideal for training diffusion models. The full dataset consists of 15,000 images, but this project uses a smaller subset to make the training process more efficient while still providing high-quality outputs.

- **Domain**: Cats
- **Image Resolution**: 512 x 512 pixels
- **Training Set Size**: Approx. 5,000 images

This dataset is ideal for experimenting with generative models like DDPM, as it contains a variety of image styles and textures that allow the model to learn complex patterns during the training process.
