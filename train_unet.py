import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# paths
IMAGE_DIR = r"C:\Users\DELL\Desktop\hack\train\Color_Images"
MASK_DIR = r"C:\Users\DELL\Desktop\hack\train\Segmentation"
# dataset
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = A.Compose([
            A.Resize(256,256),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        aug = self.transform(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"].long()

        return image, mask

# dataset
dataset = SegDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

device = "cpu"
model.to(device)

# loss + optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training
for epoch in range(5):
    print("Epoch:", epoch)

    for images, masks in loader:
        images = images.to(device).float()
        masks = masks.unsqueeze(1).to(device).float()

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Loss:", loss.item())

# save
torch.save(model.state_dict(), "unet_model.pth")

print("Training finished")