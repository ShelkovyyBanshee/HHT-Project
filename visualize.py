import os
import matplotlib.pyplot as plt
import cv2
from datasets.dataset import OilSpillDataset
from utils.xml_parser import parse_xml
import numpy as np

# Пути
images_dir = "data/images"
annotations_dir = "data/annotations"
tab_path = "data/DARTIS_2019.tab"

# Dataset
dataset = OilSpillDataset(
    images_dir=images_dir,
    annotations_dir=annotations_dir,
    tab_path=tab_path,
    img_size=640
)

# Можно менять индекс для просмотра разных изображений
idx = 3000
image_tensor, mask_tensor, label, image_name = dataset[idx]


# В numpy
image_full = image_tensor.permute(1, 2, 0).numpy()  # RGB + HHT
mask = mask_tensor.squeeze(0).numpy()

# Для визуализации — только RGB
image = image_full[:, :, :3]

# Путь к XML
xml_path = os.path.join(
    annotations_dir,
    image_name.replace(".jpg", ".xml")
)

# Проверяем есть ли XML
boxes = []
if os.path.exists(xml_path):
    boxes, _ = parse_xml(xml_path)

# Для bbox берем RGB копию
image_bbox = (image.copy() * 255).astype(np.uint8)

for (xmin, ymin, xmax, ymax) in boxes:
    cv2.rectangle(image_bbox, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

# Названия классов (для читаемости)
label_map = {
    0: "oc (oil/coast)",
    1: "ow (oil/water)",
    2: "nc (no_oil/coast)",
    3: "nw (no_oil/water)"
}

label_name = label_map.get(label.item(), "unknown")

# Визуализация
plt.figure(figsize=(15, 5))

# 1. Оригинал
plt.subplot(1, 3, 1)
plt.title(f"Image\n{image_name}\nLabel: {label_name}")
plt.imshow(image)
plt.axis("off")

# 2. Bounding boxes (если есть)
plt.subplot(1, 3, 2)
if len(boxes) > 0:
    plt.title(f"Bounding Boxes ({len(boxes)})")
    plt.imshow(image_bbox)
else:
    plt.title("No oil objects (no XML)")
    plt.imshow(image)
plt.axis("off")

# 3. Маска
plt.subplot(1, 3, 3)
plt.title("Pseudo-Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()