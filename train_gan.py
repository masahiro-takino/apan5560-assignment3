import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.models.gan import Generator, Discriminator


def train_gan(
    epochs: int = 2,
    batch_size: int = 128,
    noise_dim: int = 100,
    lr: float = 0.0002,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            optimizer_d.zero_grad()

            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            z = torch.randn(current_batch_size, noise_dim, device=device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            z = torch.randn(current_batch_size, noise_dim, device=device)
            generated_images = generator(z)
            outputs = discriminator(generated_images)

            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {avg_d_loss:.4f}, "
            f"G Loss: {avg_g_loss:.4f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/gan_generator.pth")
    torch.save(discriminator.state_dict(), "models/gan_discriminator.pth")

    print("Saved models to models/ directory.")


if __name__ == "__main__":
    train_gan()
