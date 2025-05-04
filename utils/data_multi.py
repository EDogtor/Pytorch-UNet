import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

# 创建病变掩膜的函数
def create_mask(image_size, x, y, mask_size=200):
    """
    根据病变坐标(x, y)创建一个病变掩膜，大小为mask_size。
    """
    mask = np.zeros(image_size, dtype=np.float32)
    
    # 计算窗口的边界
    xmin = max(x - mask_size // 2, 0)
    xmax = min(x + mask_size // 2, image_size[0])
    ymin = max(y - mask_size // 2, 0)
    ymax = min(y + mask_size // 2, image_size[1])
    
    mask[ymin:ymax, xmin:xmax] = 1  # 设定病变区域为1
    
    return mask

class MultiModalDataset(Dataset):
    def __init__(self, image_paths, patient_data, crop_size: int = 200, transform=None):
        self.image_paths = list(map(str, image_paths))  # 将路径转换为字符串列表
        self.transform = transform
        self.crop_size = crop_size

        # 规范化列名
        pdf = patient_data.copy()
        pdf.columns = [c.strip().lower() for c in pdf.columns]

        col_id = next((c for c in pdf.columns if c in {"rider_id", "rider-id", "patient_id"}), None)
        col_x = next((c for c in pdf.columns if c.startswith("x")), None)
        col_y = next((c for c in pdf.columns if c.startswith("y")), None)

        if col_id is None or col_x is None or col_y is None:
            raise ValueError("patient_data must have ID, x, y columns (case-insensitive)")

        # 构建 ID -> (x, y) 查找字典
        self.coord_lookup = {
            str(row[col_id]): (int(row[col_x]), int(row[col_y])) for _, row in pdf.iterrows()
        }

    # 返回数据集的长度
    def __len__(self):
        return len(self.image_paths)

    # 获取特定索引的数据
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = Path(img_path).stem
        patient_id = stem.split("_")[0]

        if patient_id not in self.coord_lookup:
            raise KeyError(f"No coordinates for patient ID '{patient_id}' in CSV.")
        x, y = self.coord_lookup[patient_id]

        image = Image.open(img_path).convert("RGB")
        mask = create_mask(image.size, x, y, self.crop_size)

        if self.transform:
            image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32)

        return {"image": image, "mask": mask}
