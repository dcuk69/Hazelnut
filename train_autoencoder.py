import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Defaults (kept as fallbacks)
DATA_PATH = os.path.dirname(__file__)
CATEGORY = 'hazelnut'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_dataloaders(data_path, category, img_size, batch_size, use_random=False):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_path, category, 'train')
    test_path = os.path.join(data_path, category, 'test')

    if not use_random and os.path.isdir(train_path) and any(os.scandir(train_path)):
        train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Using real training data: {train_path} ({len(train_dataset)} samples)")
    else:
        # fallback to random data for quick tests
        total = batch_size * 8
        data = torch.rand(total, 3, img_size, img_size)
        train_dataset = TensorDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print(f"Using random training data ({total} samples)")

    if not use_random and os.path.isdir(test_path) and any(os.scandir(test_path)):
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        print(f"Using real test data: {test_path} ({len(test_dataset)} samples)")
    else:
        # small random test set
        data = torch.rand(9, 3, img_size, img_size)
        test_dataset = TensorDataset(data)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        print("Using random test data (9 samples)")

    return train_loader, test_loader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_path, args.category, args.img_size, args.batch_size, args.use_random)

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    loss_history = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}")

    print("Training complete!")

    # Visualization (only show/save first 3)
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    samples_shown = 0

    with torch.no_grad():
        for batch in test_loader:
            if samples_shown >= 3:
                break
            images = batch[0].to(device)
            reconstructions = model(images)

            diff = (images - reconstructions).pow(2).mean(dim=1, keepdim=True)

            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            rec_np = reconstructions[0].cpu().permute(1, 2, 0).numpy()
            diff_np = diff[0].cpu().squeeze().numpy()
            diff_np = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-5)

            axes[samples_shown, 0].imshow(img_np)
            axes[samples_shown, 0].set_title("Input Image")
            axes[samples_shown, 0].axis('off')

            axes[samples_shown, 1].imshow(rec_np)
            axes[samples_shown, 1].set_title("Reconstruction")
            axes[samples_shown, 1].axis('off')

            axes[samples_shown, 2].imshow(diff_np, cmap='jet')
            axes[samples_shown, 2].set_title("Anomaly Map (Error)")
            axes[samples_shown, 2].axis('off')

            samples_shown += 1

    plt.tight_layout()
    plt.savefig('result_grid.png', dpi=150, bbox_inches='tight')
    print("Results saved to result_grid.png")

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    print("Loss curve saved to loss_curve.png")


def parse_args():
    parser = argparse.ArgumentParser(description='Train convolutional autoencoder')
    parser.add_argument('--data-path', default=DATA_PATH)
    parser.add_argument('--category', default=CATEGORY)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, dest='lr', default=LEARNING_RATE)
    parser.add_argument('--use-random', action='store_true', help='Use random tensors instead of reading dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
