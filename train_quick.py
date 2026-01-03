import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset

# Quick training config
DATA_PATH = r'C:\Users\dcuk1\OneDrive\Documents\dataset'
CATEGORY = 'hazelnut'
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-3

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Simple AE (same as train_autoencoder)
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

# Prepare dataset: use real images if available, else random tensors
train_path = os.path.join(DATA_PATH, CATEGORY, 'train')
use_real = os.path.isdir(train_path) and any(os.scandir(train_path))

transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

if use_real:
    print('Using real dataset at', train_path)
    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
else:
    print('Real dataset not found or empty; using random data')
    total = 64
    data = torch.rand(total, 3, IMG_SIZE, IMG_SIZE)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, opt
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# One-epoch training
model.train()
for epoch in range(EPOCHS):
    running = 0.0
    count = 0
    for batch in loader:
        # batch could be (images, labels) or (images,)
        images = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running += loss.item()
        count += 1
    avg = running / max(1, count)
    print(f'Epoch {epoch+1}/{EPOCHS} - avg loss: {avg:.6f}')

print('Quick train test complete')
