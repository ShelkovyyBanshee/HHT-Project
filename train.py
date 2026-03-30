import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

from datasets.dataset import OilSpillDataset
from models.unet_resnet34 import UNetResNet34
from losses.losses import CombinedLoss

from torch.cuda.amp import autocast, GradScaler

# ===== ПАРАМЕТРЫ =====
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
HHT_IMFS = 2

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ===== DATA =====
dataset = OilSpillDataset(
    images_dir="data/images",
    annotations_dir="data/annotations",
    tab_path="data/DARTIS_2019.tab",
    img_size=IMG_SIZE,
    hht_imfs=HHT_IMFS
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ===== MODEL =====
model = UNetResNet34(in_channels=3 + HHT_IMFS).to(device)

# ===== LOSS =====
criterion = CombinedLoss()

# ===== OPTIMIZER =====
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ===== AMP =====
scaler = GradScaler()

# ===== SAVE DIR =====
os.makedirs("checkpoints", exist_ok=True)


# ===== FUNCTION: SMOOTH MASK =====
def smooth_mask(mask_tensor):
    """
    mask_tensor: [B, 1, H, W]
    возвращает сглаженную маску (soft labels)
    """
    mask_np = mask_tensor.cpu().numpy()

    smoothed = []
    for m in mask_np:
        m = m.squeeze(0)

        # Gaussian blur
        m = cv2.GaussianBlur(m, (11, 11), sigmaX=5)

        # нормализация
        m = m / (m.max() + 1e-8)

        smoothed.append(m)

    smoothed = np.stack(smoothed)
    smoothed = torch.from_numpy(smoothed).unsqueeze(1).float()

    return smoothed.to(mask_tensor.device)


# ===== TRAIN =====
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()

    epoch_loss = 0
    epoch_seg = 0
    epoch_clf = 0

    for images, masks, labels, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # 🔥 СГЛАЖИВАЕМ МАСКИ
        masks = smooth_mask(masks)

        optimizer.zero_grad()

        with autocast():
            seg_out, clf_out = model(images)
            loss, seg_loss, clf_loss = criterion(seg_out, masks, clf_out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_seg += seg_loss.item()
        epoch_clf += clf_loss.item()

    # усреднение
    epoch_loss /= len(loader)
    epoch_seg /= len(loader)
    epoch_clf /= len(loader)

    print(f"Epoch {epoch}: total={epoch_loss:.4f}, seg={epoch_seg:.4f}, clf={epoch_clf:.4f}")

    # ===== SAVE BEST =====
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")

    # ===== SAVE LAST =====
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")