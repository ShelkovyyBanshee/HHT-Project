import torch
from torch.utils.data import DataLoader

from datasets.dataset import OilSpillDataset
from models.unet_resnet34 import UNetResNet34  # путь под себя

# ===== ПАРАМЕТРЫ =====
IMG_SIZE = 256
BATCH_SIZE = 2
HHT_IMFS = 2

# ===== DATASET =====
dataset = OilSpillDataset(
    images_dir="data/images",
    annotations_dir="data/annotations",
    tab_path="data/DARTIS_2019.tab",
    img_size=IMG_SIZE,
    hht_imfs=HHT_IMFS
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== DEVICE =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ===== MODEL =====
model = UNetResNet34(in_channels=3 + HHT_IMFS).to(device)
model.eval()

# ===== ТЕСТ =====
with torch.no_grad():
    images, masks, labels, names = next(iter(loader))

    print("Input image shape:", images.shape)  # [B, C, H, W]
    print("Mask shape:", masks.shape)  # [B, 1, H, W]
    print("Labels shape:", labels.shape)  # [B]

    images = images.to(device)

    seg_out, clf_out = model(images)

    print("Segmentation output:", seg_out.shape)  # [B, 1, H, W]
    print("Classification output:", clf_out.shape)  # [B, 4]

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")