import os
from torch import Tensor
from torch.utils.data import Dataset
from utils import resolve_path
from imgproc import image_to_tensor, image_resize  
from typing import List, Callable, Optional
import numpy as np
import cv2

class OCTImageDataset(Dataset):
    """A medical OCT dataset that returns 128x128 GT images and 32x32 LR images."""

    def __init__(
            self,
            images_root: str,
            gt_image_size: int = 128,
            upscale_factor: int = 4,
            transform: Optional[Callable] = None
    ) -> None:
        super(OCTImageDataset, self).__init__()

        images_root = resolve_path(images_root)
        if not os.path.isdir(images_root):
            raise FileNotFoundError(f"OCT image root does not exist: {images_root}")

        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.image_paths = self._collect_image_paths(images_root)
        self.transform = transform

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in OCT image root: {images_root}")

    @staticmethod
    def _collect_image_paths(images_root: str) -> List[str]:
        image_paths = []
        for root, _, files in os.walk(images_root):
            if os.path.basename(root).startswith("__MACOSX"):
                continue
            for file_name in sorted(files):
                lower_name = file_name.lower()
                if lower_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                    image_paths.append(os.path.join(root, file_name))
        return image_paths

    @staticmethod
    def _prepare_image(image: np.ndarray, target_size: int) -> np.ndarray:
        height, width = image.shape[:2]
        if height == target_size and width == target_size:
            return image

        if height < target_size or width < target_size:
            return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        top = (height - target_size) // 2
        left = (width - target_size) // 2
        image = image[top:top + target_size, left:left + target_size]
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        return image

    def __getitem__(self, batch_index: int) -> dict:
        image_path = self.image_paths[batch_index]
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
        image = self._prepare_image(image, self.gt_image_size)
        gt_tensor = image_to_tensor(image, False, False)
        lr_tensor = image_resize(gt_tensor, 1.0 / self.upscale_factor)

        if self.transform:
            gt_tensor = self.transform(gt_tensor)
            lr_tensor = self.transform(lr_tensor)

        return {
            "gt": gt_tensor,
            "lr": lr_tensor,
            "image_name": image_path,
        }

    def __len__(self) -> int:
        return len(self.image_paths)