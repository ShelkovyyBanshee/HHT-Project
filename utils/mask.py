import numpy as np


def create_mask(boxes, shape):
    mask = np.zeros(shape, dtype=np.uint8)

    for (xmin, ymin, xmax, ymax) in boxes:
        mask[ymin:ymax, xmin:xmax] = 1

    return mask