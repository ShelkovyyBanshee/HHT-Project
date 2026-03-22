import numpy as np
import cv2
from scipy.signal import hilbert


class HHTProcessor:
    def __init__(self, num_imfs=2, resize=None):
        """
        num_imfs : сколько IMF-подобных каналов использовать
        resize : уменьшение размера для ускорения вычислений (например 128)
        """
        self.num_imfs = num_imfs
        self.resize = resize

    def _normalize(self, x):
        x = x - np.min(x)
        x = x / (np.max(x) + 1e-8)
        return x

    def _compute_imfs(self, gray):
        """
        Приближение IMF через Hilbert: берем амплитуду аналитического сигнала по строкам
        """
        H, W = gray.shape
        imf_maps = []

        for k in range(self.num_imfs):
            imf = np.zeros((H, W), dtype=np.float32)
            for i in range(H):
                analytic_signal = hilbert(gray[i, :])
                amplitude_envelope = np.abs(analytic_signal)
                # Сдвигаем канал для k-го IMF-подобного компонента
                imf[i, :] = np.roll(amplitude_envelope, k * 2)
            imf_maps.append(self._normalize(imf))
        return imf_maps

    def process(self, image):
        """
        image: numpy (H, W, 3), uint8
        return: list of IMF maps (H, W), float32
        """
        orig_h, orig_w = image.shape[:2]

        if self.resize is not None:
            image_resized = cv2.resize(image, (self.resize, self.resize))
        else:
            image_resized = image

        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)
        imf_maps = self._compute_imfs(gray)

        # приводим обратно к оригинальному размеру
        imf_maps_resized = [cv2.resize(imf, (orig_w, orig_h)) for imf in imf_maps]

        return imf_maps_resized