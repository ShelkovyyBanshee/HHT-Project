import os
import sys
import traceback

import cv2
import numpy as np
import torch

from PyQt5 import uic
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from models.unet_resnet34 import UNetResNet34
from utils.hht import HHTProcessor


IMG_SIZE = 640
HHT_IMFS = 2
USE_HHT = True
MODEL_PATH = "checkpoints/best_model.pth"

CLASS_MAP = {
    0: "oc (oil/coast)",
    1: "ow (oil/water)",
    2: "nc (no_oil/coast)",
    3: "nw (no_oil/water)",
}

NO_OIL_CLASSES = {2, 3}

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    in_channels = 3 + HHT_IMFS if USE_HHT else 3
    model = UNetResNet34(in_channels=in_channels).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


model = build_model()
hht_processor = HHTProcessor(num_imfs=HHT_IMFS, resize=128) if USE_HHT else None


def safe_imread(image_path: str):
    file_bytes = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    return image


def preprocess_image(image_path: str):
    image_bgr = safe_imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

    if USE_HHT:
        imf_maps = hht_processor.process(image_rgb)
        hht_stack = np.stack(imf_maps, axis=-1)
        hht_stack = (hht_stack * 255).astype(np.uint8)
        image_full = np.concatenate([image_rgb, hht_stack], axis=-1)
    else:
        image_full = image_rgb

    image_tensor = torch.from_numpy(image_full).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_rgb, image_tensor


def postprocess_mask(mask_probs: np.ndarray):
    mask = (mask_probs > 0.5).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_bboxes_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))

    return boxes


def render_result_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    boxes,
    class_name: str,
    show_segmentation: bool = True
):
    image_draw = image_rgb.copy()

    if show_segmentation:
        colored_mask = np.zeros_like(image_draw)
        colored_mask[:, :, 0] = mask * 255
        image_draw = cv2.addWeighted(image_draw, 0.7, colored_mask, 0.3, 0)

        for (xmin, ymin, xmax, ymax) in boxes:
            cv2.rectangle(image_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.putText(
        image_draw,
        f"Class: {class_name}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return image_draw


def np_to_qpixmap(image_rgb: np.ndarray):
    image_rgb = np.ascontiguousarray(image_rgb)
    h, w, ch = image_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class AnalysisWorker(QObject):
    progress_changed = pyqtSignal(int)
    result_ready = pyqtSignal(str, str, QPixmap)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit()
                return

            for idx, path in enumerate(self.file_paths, start=1):
                if not self._is_running:
                    break

                image_rgb, tensor = preprocess_image(path)

                with torch.no_grad():
                    seg_out, clf_out = model(tensor)

                mask_probs = torch.sigmoid(seg_out).cpu().numpy()[0, 0]
                mask = postprocess_mask(mask_probs)

                pred_class_idx = torch.argmax(clf_out, dim=1).item()
                class_name = CLASS_MAP.get(pred_class_idx, "unknown")

                if pred_class_idx in NO_OIL_CLASSES:
                    boxes = []
                    show_segmentation = False
                else:
                    boxes = get_bboxes_from_mask(mask)
                    show_segmentation = True

                rendered = render_result_image(
                    image_rgb=image_rgb,
                    mask=mask,
                    boxes=boxes,
                    class_name=class_name,
                    show_segmentation=show_segmentation
                )
                pixmap = np_to_qpixmap(rendered)

                self.result_ready.emit(path, class_name, pixmap)

                progress = int(idx / total * 100)
                self.progress_changed.emit(progress)

            self.finished.emit()

        except Exception:
            self.error_occurred.emit(traceback.format_exc())
            self.finished.emit()

class DropFrame(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            files = [url.toLocalFile() for url in event.mimeData().urls()]
            if any(self._is_image_file(f) for f in files):
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        files = [f for f in files if self._is_image_file(f)]
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _is_image_file(path):
        ext = os.path.splitext(path)[1].lower()
        return ext in [".jpg", ".jpeg", ".png"]


class ResultCard(QWidget):
    def __init__(self, file_path: str, class_name: str, pixmap: QPixmap, parent=None):
        super().__init__(parent)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        self.setStyleSheet("""
            QWidget {
                background: white;
                border: 1px solid #cfcfcf;
                border-radius: 6px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)

        file_name = os.path.basename(file_path)

        self.title_label = QLabel(f"Файл: {file_name}")
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold;")

        self.class_label = QLabel(f"Классификация: {class_name}")
        self.class_label.setStyleSheet("font-size: 26px;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Было 900, стало 600: уменьшение примерно в 1.5 раза
        scaled = pixmap.scaledToWidth(600, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedHeight(scaled.height() + 10)

        root_layout.addWidget(self.title_label)
        root_layout.addWidget(self.class_label)
        root_layout.addWidget(self.image_label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main_window.ui", self)

        self._worker_thread = None
        self._worker = None

        self.progressBar.hide()
        self.progressBar.setValue(0)
        self.statusLabel.setText("")

        self.openFilesButton.clicked.connect(self.select_files)
        self.clearButton.clicked.connect(self.clear_results)

        self._replace_drop_frame()

    def _set_status(self, text: str):
        self.statusLabel.setText(text)

    def _replace_drop_frame(self):
        parent_layout = self.verticalLayout_main
        old_widget = self.dropFrame
        index = parent_layout.indexOf(old_widget)

        self.dropFrameNew = DropFrame(self.centralwidget)
        self.dropFrameNew.setObjectName("dropFrame")
        self.dropFrameNew.setMinimumHeight(110)
        self.dropFrameNew.setStyleSheet("""
            QFrame#dropFrame {
                border: 2px dashed #888;
                border-radius: 8px;
                background: #f7f7f7;
            }
        """)

        layout = QVBoxLayout(self.dropFrameNew)
        layout.setContentsMargins(8, 8, 8, 8)

        self.dropLabelNew = QLabel(
            "Перетащите сюда .jpg/.jpeg/.png изображения\nили нажмите «Выбрать изображения»"
        )
        self.dropLabelNew.setAlignment(Qt.AlignCenter)
        self.dropLabelNew.setStyleSheet("font-size: 14px; color: #444;")
        layout.addWidget(self.dropLabelNew)

        parent_layout.removeWidget(old_widget)
        old_widget.deleteLater()
        parent_layout.insertWidget(index, self.dropFrameNew)

        self.dropFrameNew.files_dropped.connect(self.start_analysis)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбрать изображения",
            "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if files:
            self.start_analysis(files)

    def clear_results(self):
        layout = self.resultsLayout

        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                layout.removeWidget(widget)
                widget.deleteLater()

        self.progressBar.hide()
        self.progressBar.setValue(0)
        self._set_status("Результаты очищены")

    def start_analysis(self, files):
        if not files:
            return

        files = self._normalize_and_filter_files(files)
        if not files:
            QMessageBox.warning(self, "Предупреждение", "Не найдено подходящих изображений.")
            return

        self.progressBar.show()
        self.progressBar.setValue(0)
        self._set_status(f"Запущен анализ: {len(files)} изображ.")

        self.openFilesButton.setEnabled(False)
        self.clearButton.setEnabled(False)

        self._worker_thread = QThread()
        self._worker = AnalysisWorker(files)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self.progressBar.setValue)
        self._worker.result_ready.connect(self.add_result_card)
        self._worker.error_occurred.connect(self.show_error)
        self._worker.finished.connect(self.analysis_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._worker_thread.start()

    def _normalize_and_filter_files(self, files):
        valid = []
        seen = set()

        for f in files:
            f = os.path.normpath(f)
            if not os.path.isfile(f):
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue

            if f not in seen:
                valid.append(f)
                seen.add(f)

        return valid

    def add_result_card(self, file_path, class_name, pixmap):
        insert_index = max(0, self.resultsLayout.count() - 1)
        card = ResultCard(file_path, class_name, pixmap, self)
        self.resultsLayout.insertWidget(insert_index, card)

    def show_error(self, error_text):
        self._set_status("Ошибка анализа")
        QMessageBox.critical(self, "Ошибка анализа", error_text)

    def analysis_finished(self):
        self.progressBar.setValue(100)
        self.openFilesButton.setEnabled(True)
        self.clearButton.setEnabled(True)
        self._set_status("Анализ завершён")

    def closeEvent(self, event):
        try:
            if self._worker is not None:
                self._worker.stop()
            if self._worker_thread is not None and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()