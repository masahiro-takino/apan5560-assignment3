import base64
import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from torchvision.utils import make_grid

from app.models.gan import Generator


app = FastAPI(title="Assignment 3: GAN Image Generation API")

NOISE_DIM = 100
MODEL_PATH = Path("models/gan_generator.pth")


def load_generator() -> Generator:
    model = Generator(noise_dim=NOISE_DIM)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained generator model not found. Please run train_gan.py first."
        )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@app.get("/")
def root():
    return {
        "message": "Assignment 3 GAN Image Generation API is running.",
        "endpoints": ["/health", "/generate"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/generate")
def generate(num_images: int = 16):
    if num_images < 1 or num_images > 64:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 64.")

    try:
        generator = load_generator()
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    with torch.no_grad():
        z = torch.randn(num_images, NOISE_DIM)
        generated = generator(z)

    generated = (generated + 1.0) / 2.0
    grid = make_grid(generated, nrow=4)

    image_tensor = grid.squeeze(0).numpy()
    image_array = (image_tensor * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "num_images": num_images,
        "image_format": "png",
        "image_base64": encoded_image,
    }
