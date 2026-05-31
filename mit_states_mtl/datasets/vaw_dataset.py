"""
VAW (Visual Attributes in the Wild) 数据集加载器（对齐论文版）
=============================================================
关键点（与 v1 相比）：
  1. **物体名是 INPUT**，不再当成分类输出。每条样本返回 object_id（int），
     由模型 embedding 后调制图像特征。
  2. **属性是多标签**，y_c ∈ {1, 0, -1}：
        +1 = 显式正例
         0 = 显式负例
        -1 = 缺失（论文用"软负样本"处理）
  3. **保留所有显式标注**：同时使用 positive_attributes 和 negative_attributes。
  4. **bbox 外扩**：宽高分别加 `min(w,h) × 0.3`（论文做法）。
"""

import json
import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── 颜色属性集合（用于 random grayscale 增强：仅当样本没有任何颜色属性时启用）──
COLOR_ATTRS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "tan", "beige",
    "turquoise", "cyan", "magenta", "gold", "silver", "colorful",
    "multicolored", "dark", "light", "bright", "pale",
}


def _find_image(image_id: int, img_dirs: List[str]) -> Optional[str]:
    fname = f"{image_id}.jpg"
    for d in img_dirs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def build_vocab(
    train_json: str,
    min_obj_count: int = 30,
    min_attr_count: int = 50,
):
    """
    扫描训练集构建词典。
      - 物体：出现次数 ≥ min_obj_count 的全部保留
      - 属性：作为正例或负例至少出现 min_attr_count 次的全部保留
    返回 (obj2id, attr2id)。
    """
    with open(train_json, encoding="utf-8") as f:
        data = json.load(f)

    obj_cnt  = Counter(d["object_name"] for d in data)
    attr_cnt = Counter()
    for d in data:
        attr_cnt.update(d.get("positive_attributes", []))
        attr_cnt.update(d.get("negative_attributes", []))

    obj2id  = {o: i for i, (o, c) in enumerate(obj_cnt.most_common())
               if c >= min_obj_count}
    attr2id = {a: i for i, (a, c) in enumerate(attr_cnt.most_common())
               if c >= min_attr_count}
    return obj2id, attr2id


def compute_attr_stats(ann_path: str, attr2id: dict):
    """统计每个属性在训练集中的正例/负例数量，用于 RW-BCE 重加权。"""
    n_attr = len(attr2id)
    n_pos = np.zeros(n_attr, dtype=np.int64)
    n_neg = np.zeros(n_attr, dtype=np.int64)

    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    for d in data:
        for a in d.get("positive_attributes", []):
            if a in attr2id:
                n_pos[attr2id[a]] += 1
        for a in d.get("negative_attributes", []):
            if a in attr2id:
                n_neg[attr2id[a]] += 1

    return n_pos, n_neg


class VAWDataset(Dataset):
    """
    返回
    ----
    img       : Tensor [3, H, W]
    obj_id    : int
    attr_y    : Tensor [n_attr]，y ∈ {1, 0, -1}（float32）
    """

    def __init__(
        self,
        ann_json: str,
        img_dirs: List[str],
        obj2id:  dict,
        attr2id: dict,
        img_size: int   = 224,
        pad_ratio: float = 0.3,    # 论文 min(w,h) × 0.3
        augment: bool   = False,
    ):
        with open(ann_json, encoding="utf-8") as f:
            raw = json.load(f)

        self.img_dirs  = img_dirs
        self.obj2id    = obj2id
        self.attr2id   = attr2id
        self.n_obj     = len(obj2id)
        self.n_attr    = len(attr2id)
        self.pad_ratio = pad_ratio
        self.augment   = augment
        self.img_size  = img_size

        # 只保留词典内的物体
        self.data = [d for d in raw if d["object_name"] in obj2id]

        self.normalize = T.Normalize(IMG_MEAN, IMG_STD)
        if augment:
            self.geom = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
            ])
            self.color = T.ColorJitter(0.2, 0.2, 0.2, 0.03)
        else:
            self.geom = T.Resize((img_size, img_size))
            self.color = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]

        img_path = _find_image(ann["image_id"], self.img_dirs)
        if img_path is None:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = Image.open(img_path).convert("RGB")

        # ── bbox 外扩 min(w,h) × pad_ratio（论文做法）─────────────
        W, H = img.size
        x, y, w, h = ann["instance_bbox"]
        pad = min(w, h) * self.pad_ratio
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)
        img = img.crop((x1, y1, x2, y2))

        # ── 几何增强 + 可选颜色增强 ────────────────────────────────
        img = self.geom(img)
        if self.color is not None:
            img = self.color(img)
            # 若该实例没标注任何颜色属性 → 随机灰度（论文做法）
            pos_attrs = set(ann.get("positive_attributes", []))
            if not (pos_attrs & COLOR_ATTRS) and np.random.rand() < 0.2:
                img = T.functional.rgb_to_grayscale(img, num_output_channels=3)

        # 返回 uint8 CHW 张量（不在 CPU 上归一化）。
        # float 化 + 归一化挪到 GPU 上做：共享内存传输量降到 1/4，数值不变。
        arr   = np.array(img, dtype=np.uint8)               # HWC（copy → 可写）
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8

        obj_id = self.obj2id[ann["object_name"]]

        # ── 属性标签：1 / 0 / -1（缺失）─────────────────────────────
        attr_y = torch.full((self.n_attr,), -1.0, dtype=torch.float32)
        for a in ann.get("positive_attributes", []):
            if a in self.attr2id:
                attr_y[self.attr2id[a]] = 1.0
        for a in ann.get("negative_attributes", []):
            if a in self.attr2id:
                attr_y[self.attr2id[a]] = 0.0

        return img_t, obj_id, attr_y

    @property
    def id2obj(self):
        return {v: k for k, v in self.obj2id.items()}

    @property
    def id2attr(self):
        return {v: k for k, v in self.attr2id.items()}
