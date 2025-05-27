import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import OpenEXR
import Imath
import cv2

__all__ = ['ImageDataset', 'MaskedImageDataset']

class ImageDataset(Dataset):
    """Random pixel sampler for a single image.

    Supports typical formats (PNG, JPG) as well as HDR .exr images.
    Returns `num_samples` random (x, y) pixel coordinates and their RGB
    colors, both normalised to the range [-1, 1] / [0, 1].
    """

    def __init__(self, image_path: str, size: int = 100, num_samples: int = 10_000):
        super().__init__()
        self.image_path = image_path
        self.size = size            # number of iterations per epoch
        self.num_samples = num_samples

        # ------------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------------
        if image_path.lower().endswith('.exr'):
            exr_file = OpenEXR.InputFile(image_path)
            header = exr_file.header()
            dw = header['dataWindow']
            self.width  = dw.max.x - dw.min.x + 1
            self.height = dw.max.y - dw.min.y + 1

            channels = ['R', 'G', 'B']
            img = np.zeros((3, self.height, self.width), dtype=np.float32)
            for i, ch in enumerate(channels):
                raw = exr_file.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                img[i] = np.frombuffer(raw, dtype=np.float32).reshape(self.height, self.width)
            self.image = img * 255.0  # bring into 0-255 range for parity with LDR
        else:
            img = Image.open(image_path).convert('RGB')
            self.width, self.height = img.size
            self.image = np.asarray(img).transpose(2, 0, 1).astype(np.float32)  # (3,H,W)

        # Normalise to [0,1]
        self.image /= 255.0

    # ------------------------------------------------------------------
    def __len__(self):
        return self.size

    def __getitem__(self, _):
        # Random sample pixel indices
        xs = np.random.randint(0, self.width,  self.num_samples)
        ys = np.random.randint(0, self.height, self.num_samples)

        # Normalised pixel coordinates in [-1, 1]
        coords = np.stack([
            (xs / (self.width  - 1)) * 2 - 1,
            (ys / (self.height - 1)) * 2 - 1
        ], axis=-1).astype(np.float32)  # (N,2)

        rgbs = self.image[:, ys, xs].T.astype(np.float32)  # (N,3)

        return {
            'coords': coords,
            'rgbs': rgbs
        }

# --------------------------------------------------------------------------
# Optional dataset that avoids masked regions
# --------------------------------------------------------------------------
class MaskedImageDataset(Dataset):
    def __init__(self, image_path: str, size: int = 500, num_samples: int = 10_000,
                 num_masks: int = 20, mask_size: int = 500):
        super().__init__()
        self.image_ds = ImageDataset(image_path, size=size, num_samples=num_samples)
        self.num_masks = num_masks
        self.mask_size = mask_size

        self.mask = self._random_mask(self.image_ds.height, self.image_ds.width)
        self.valid_y, self.valid_x = np.where(self.mask)

    def _random_mask(self, h, w):
        mask = np.ones((h, w), dtype=bool)
        for _ in range(self.num_masks):
            top = np.random.randint(0, h - self.mask_size)
            left = np.random.randint(0, w - self.mask_size)
            mask[top:top + self.mask_size, left:left + self.mask_size] = False
        return mask

    def __len__(self):
        return self.image_ds.size

    def __getitem__(self, _):
        indices = np.random.randint(0, len(self.valid_y), self.image_ds.num_samples)
        ys = self.valid_y[indices]
        xs = self.valid_x[indices]

        coords = np.stack([
            (xs / (self.image_ds.width  - 1)) * 2 - 1,
            (ys / (self.image_ds.height - 1)) * 2 - 1
        ], axis=-1).astype(np.float32)

        rgbs = self.image_ds.image[:, ys, xs].T.astype(np.float32)

        return {
            'coords': coords,
            'rgbs': rgbs
        } 