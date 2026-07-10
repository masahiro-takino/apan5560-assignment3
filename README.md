# Assignment 3: Image Generation

This repository contains my implementation for Assignment 3 of APANPS5560 Applied Generative AI.

The project implements a Generative Adversarial Network (GAN) in PyTorch to generate MNIST-style handwritten digit images. The trained Generator is deployed through a FastAPI application and containerized using Docker.

## Project Components

- PyTorch Generator model
- PyTorch Discriminator model
- MNIST training script
- Saved trained Generator weights
- FastAPI image generation endpoint
- Docker deployment

## GAN Architecture

The Generator takes a random noise vector of shape `(batch_size, 100)` and generates a `1 × 28 × 28` image.

The Discriminator takes a `1 × 28 × 28` image and predicts whether the image is real or generated.

## API Endpoints

### Root

```text
GET /
```

### Health Check

```text
GET /health
```

Expected response:

```json
{"status": "ok"}
```

### Image Generation

```text
GET /generate?num_images=16
```

This endpoint generates MNIST-style handwritten digit images and returns the generated image grid as a base64-encoded PNG.

## How to Train

```bash
python train_gan.py
```

## How to Run with Docker

```bash
docker build -t assignment3-gan .
docker run -p 8000:8000 assignment3-gan
```

Then open:

```text
http://localhost:8000/docs
```
