import sys
import torch
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage

from models.unet_resnet34 import UNetResNet34
from utils.hht import HHTProcessor


# ===== ПАРАМЕТРЫ =====
IMG_SIZE = 256
HHT_IMFS = 2
MODEL_PATH = "checkpoints/best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== ЗАГРУЗКА МОДЕЛИ =====
model = UNetResNet34(in_channels=3 + HHT_IMFS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

hht = HHTProcessor(num_imfs=HHT_IMFS, resize=128)


# ===== ПРЕПРОЦЕССИНГ =====
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # HHT
    imf_maps = hht.process(image)
    hht_stack = np.stack(imf_maps, axis=-1)

    image_full = np.concatenate([image, (hht_stack * 255).astype(np.uint8)], axis=-1)

    image_tensor = torch.from_numpy(image_full).permute(2, 0, 1).float() / 255.0
    return image, image_tensor.unsqueeze(0).to(device)


# ===== POST-PROCESSING =====
def postprocess(mask):
    mask = (mask > 0.5).astype(np.uint8)

    # Morphology (сглаживание)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))

    return boxes


# ===== GUI =====
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Spill Detection")

        self.btn = QPushButton("Выбрать изображение")
        self.btn.clicked.connect(self.load_image)

        self.label = QLabel("Здесь будет результат")

        layout = QVBoxLayout()
        layout.addWidget(self.btn)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.jpg *.png)")
        if not path:
            return

        image_rgb, tensor = preprocess(path)

        with torch.no_grad():
            seg_out, clf_out = model(tensor)

        # ===== MASK =====
        mask = torch.sigmoid(seg_out).cpu().numpy()[0, 0]
        mask = postprocess(mask)

        # ===== BBOX =====
        boxes = get_bboxes(mask)

        # ===== CLASS =====
        cls = torch.argmax(clf_out, dim=1).item()
        class_map = ["oc", "ow", "nc", "nw"]
        cls_name = class_map[cls]

        # ===== DRAW =====
        image_draw = image_rgb.copy()

        # overlay mask
        colored_mask = np.zeros_like(image_draw)
        colored_mask[:, :, 0] = mask * 255  # красный

        image_draw = cv2.addWeighted(image_draw, 0.7, colored_mask, 0.3, 0)

        # bbox
        for (xmin, ymin, xmax, ymax) in boxes:
            cv2.rectangle(image_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # текст
        cv2.putText(image_draw, f"Class: {cls_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # convert to Qt
        h, w, ch = image_draw.shape
        bytes_per_line = ch * w
        q_img = QImage(image_draw.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(q_img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())