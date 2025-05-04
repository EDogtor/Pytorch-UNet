# utils/riderlung_dataset.py
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import torchvision.transforms as T
import random

class RiderLungDataset(Dataset):
    def __init__(self, imgs_dir: Path, masks_dir: Path, scale: float = 1.0, augment: bool = False):
        self.imgs_dir = Path(imgs_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.augment = augment

        self.ids = sorted(p.stem for p in self.imgs_dir.glob('*.png'))
        if not self.ids:
            raise RuntimeError(f'❗ 在 {self.imgs_dir} 没发现任何 PNG！')

        logging.info(f'✅ RiderLungDataset – 共 {len(self.ids)} 张切片')

        # 定义数据增强方法
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.RandomResizedCrop(256, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])

    def __len__(self):
        """返回数据集的大小"""
        return len(self.ids)

    def _mask_path(self, img_stem: str) -> Path:
        """把 ..._image_... → ..._mask_..."""
        return self.masks_dir / f"{img_stem.replace('_image_', '_mask_')}.png"
    

    def _preprocess_img(self, pil_img: Image.Image, is_mask: bool = False):
        if self.scale != 1.0:
            w, h = pil_img.size
            new_size = (int(w * self.scale), int(h * self.scale))
            pil_img = pil_img.resize(new_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)

        np_img = np.array(pil_img)

        if is_mask:
            if np_img.ndim == 3:  # Some masks are stored as RGB
                np_img = np_img[..., 0]
            np_img = (np_img > 127).astype(np.uint8)  # Ensure masks are in uint8
            return torch.from_numpy(np_img).long()  # (H, W)
        else:
            if np_img.ndim == 2:
                np_img = np_img[..., np.newaxis]
            np_img = np_img.transpose((2, 0, 1)) / 255.0  # Normalize the image

        # Convert image to float32 to ensure compatibility with PIL
            np_img = np_img.astype(np.float32)

            return torch.from_numpy(np_img).float()  # (C, H, W)


    def __getitem__(self, idx: int):
        img_stem  = self.ids[idx]
        img_path  = self.imgs_dir  / f"{img_stem}.png"
        mask_path = self._mask_path(img_stem)

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        return {
            "image": self._preprocess_img(image, is_mask=False),
            "mask" : self._preprocess_img(mask, is_mask=True),
        }