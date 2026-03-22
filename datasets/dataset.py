import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from utils.xml_parser import parse_xml
from utils.mask import create_mask
import numpy as np
import cv2

# Подключаем новый HHTProcessor
from utils.hht import HHTProcessor


class OilSpillDataset(Dataset):
    def __init__(
        self,
        images_dir,
        annotations_dir,
        tab_path,
        img_size=640,
        transform=None,
        hht_imfs=2
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.img_size = img_size
        self.transform = transform
        self.use_hht = True

        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

        self.tab_df = pd.read_csv(tab_path, sep='\t')
        self.tab_df.columns = self.tab_df.columns.str.strip()
        self.label_mapping = {"oc":0, "ow":1, "nc":2, "nw":3}

        if self.use_hht:
            # уменьшение до 128 для ускорения
            self.hht = HHTProcessor(num_imfs=hht_imfs, resize=128)

    def __len__(self):
        return len(self.image_files)

    def get_label(self, image_name):
        row = self.tab_df[self.tab_df["IMAGE (jpg_file)"] == image_name]

        if row.empty:
            return -1

        label_col = None
        for col in self.tab_df.columns:
            if col.startswith("Image set"):
                label_col = col
                break

        if label_col is None:
            raise ValueError("Column with image class not found")

        label_str = str(row.iloc[0][label_col]).strip()
        return self.label_mapping.get(label_str, -1)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        xml_name = image_name.replace(".jpg", ".xml")

        image_path = os.path.join(self.images_dir, image_name)
        image_pil = Image.open(image_path).convert("RGB")

        if self.img_size is not None:
            image_pil = image_pil.resize((self.img_size, self.img_size))

        image_np = np.array(image_pil)

        # ===== HHT =====
        if self.use_hht:
            imf_maps = self.hht.process(image_np)  # теперь точно (H, W)
            hht_stack = np.stack(imf_maps, axis=-1)  # (H, W, K)
            image_np = np.concatenate([image_np, (hht_stack * 255).astype(np.uint8)], axis=-1)

        # to tensor
        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # ===== MASK =====
        xml_path = os.path.join(self.annotations_dir, xml_name)
        if os.path.exists(xml_path):
            boxes, _ = parse_xml(xml_path)
            mask = create_mask(boxes, shape=(self.img_size, self.img_size))
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = T.ToTensor()(mask)

        # ===== LABEL =====
        label = self.get_label(image_name)
        label = torch.tensor(label, dtype=torch.long)

        return image, mask, label, image_name