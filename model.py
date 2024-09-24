import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.images = sorted(os.listdir(images_path))
        self.masks = sorted(os.listdir(masks_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def down_sample_block(in_dim, out_dim):
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                conv_block(in_dim, out_dim)
            )

        def up_sample_block(in_dim, out_dim):
            return nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = down_sample_block(64, 128)
        self.enc3 = down_sample_block(128, 256)
        self.enc4 = down_sample_block(256, 512)
        self.center = down_sample_block(512, 1024)
        self.dec4 = nn.Sequential(up_sample_block(1024, 512), conv_block(1024, 512))
        self.dec3 = nn.Sequential(up_sample_block(512, 256), conv_block(512, 256))
        self.dec2 = nn.Sequential(up_sample_block(256, 128), conv_block(256, 128))
        self.dec1 = nn.Sequential(up_sample_block(128, 64), conv_block(128, 64))
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)

        dec4 = self.dec4[0](center)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4[1](dec4)

        dec3 = self.dec3[0](dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3[1](dec3)

        dec2 = self.dec2[0](dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2[1](dec2)

        dec1 = self.dec1[0](dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1[1](dec1)

        return self.final(dec1)
    