import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def to_int64(x):
    return np.array(x).astype(np.int64)

def subtract_one(x):
    return x - 1

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.long)

#Ran with:
# PyTorch
# including torch, torchdivision, torchaudio.
# Others:
# matplotlib, numpy, tqdm, pillow



#Data from OxfordIIITPet dataset

#Can also use custom images (as is setup), jpeg or png, auto-compressed


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        logits = self.outc(x)
        return logits

IMAGE_SIZE = (128, 128)

img_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
    transforms.Lambda(to_int64),
    transforms.Lambda(subtract_one),
    transforms.Lambda(to_tensor)
])

data_root = './data'
train_dataset = OxfordIIITPet(root=data_root, split='trainval',
                              target_types='segmentation',
                              download=True,
                              transform=img_transforms,
                              target_transform=mask_transforms)
val_dataset = OxfordIIITPet(root=data_root, split='test',
                            target_types='segmentation',
                            download=True,
                            transform=img_transforms,
                            target_transform=mask_transforms)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def train_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def visualize_sample(model, device, dataloader, num_samples=3):
    model.eval()
    images, masks = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    images = images * std + mean
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))
        gt = masks[i]
        pr = preds[i]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gt, cmap="jet", vmin=0, vmax=model.n_classes-1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pr, cmap="jet", vmin=0, vmax=model.n_classes-1)
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()

def interactive_inference(model, device):
    allowed_extensions = ['png', 'jpg', 'jpeg']
    while True:
        filename = input("Enter the file name of the image (or type 'exit' to quit): ")
        if filename.lower() == 'exit':
            break
        ext = filename.split('.')[-1].lower()
        if ext not in allowed_extensions:
            print("File not supported. Please enter a PNG or JPEG image.")
            continue
        if not os.path.exists(filename):
            print("File does not exist. Try again.")
            continue
        try:
            img = Image.open(filename).convert("RGB")
        except Exception as e:
            print("Error opening image:", e)
            continue

        input_tensor = img_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = input_tensor.squeeze(0).cpu().numpy()
        for c in range(3):
            img_np[c] = img_np[c] * std[c] + mean[c]
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        pred_np = prediction.cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        axes[1].imshow(pred_np, cmap="jet", vmin=0, vmax=model.n_classes-1)
        axes[1].set_title("Predicted Segmentation")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    n_classes = 3
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    NUM_EPOCHS = 10
    best_val_loss = float('inf')
    save_path = "unet_oxford_pet.pth"
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}:")
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss = validate_epoch(model, device, val_loader, criterion)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("  Best model saved.")
    visualize_sample(model, device, val_loader, num_samples=3)
    print("Entering interactive inference mode.")
    interactive_inference(model, device)

if __name__ == '__main__':
    main()
