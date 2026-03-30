import os
import numpy as np


class HHTCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, image_name):
        base = os.path.splitext(image_name)[0]
        return os.path.join(self.cache_dir, base + ".npy")

    def exists(self, image_name):
        return os.path.exists(self.get_cache_path(image_name))

    def load(self, image_name):
        path = self.get_cache_path(image_name)
        return np.load(path)

    def save(self, image_name, data):
        path = self.get_cache_path(image_name)
        np.save(path, data)