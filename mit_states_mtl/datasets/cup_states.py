"""
杯子状态数据集（双标签 MTL 版）
================================
从 cup_dataset.csv 读取图片，将复合状态 ID 拆分为两个独立标签:
    orientation: 0=upright  1=sideways  2=inverted
    fill:        0=empty    1=half      2=full    3=N/A

复合状态到双标签映射（基于 classes_state.json）:
    0 empty    → orientation=0(upright),  fill=0(empty)
    1 half     → orientation=0(upright),  fill=1(half)
    2 full     → orientation=0(upright),  fill=2(full)
    3 sideways → orientation=1(sideways), fill=3(N/A)
    4 inverted → orientation=2(inverted), fill=3(N/A)

CSV 中 image_path 格式: cup/<state>/filename
    例: cup/half/000001.png

返回格式与 MITStatesDataset 相同:
    image [3, H, W], obj_id(orientation), state_id(fill)
"""

import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


_COMPOUND_TO_DUAL = {
    0: (0, 0),   # empty    → upright, empty
    1: (0, 1),   # half     → upright, half
    2: (0, 2),   # full     → upright, full
    3: (1, 3),   # sideways → sideways, N/A
    4: (2, 3),   # inverted → inverted, N/A
}

ORIENTATION_NAMES = ["upright", "sideways", "inverted"]
FILL_NAMES        = ["empty", "half", "full", "N/A"]


class CupDataset(Dataset):
    """
    杯子状态 MTL 数据集。

    每个样本返回:
        image          : FloatTensor [3, H, W]
        orientation_id : int  (0=upright, 1=sideways, 2=inverted)
        fill_id        : int  (0=empty, 1=half, 2=full, 3=N/A)

    Args:
        csv_path    : cup_dataset.csv 路径
        split       : "train" | "val" | "test"
        transform   : torchvision transform，None 时使用默认
        split_ratio : (train, val, test) 比例
        seed        : 随机种子
    """

    num_objects = len(ORIENTATION_NAMES)   # 3，对应 obj_head 输出
    num_states  = len(FILL_NAMES)          # 4，对应 state_head 输出

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        transform=None,
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        # datasets/ 目录，image_path 相对于此
        self.datasets_dir = os.path.dirname(os.path.abspath(csv_path))
        self.split = split

        rows = self._read_csv(csv_path)
        splits = self._stratified_split(rows, split_ratio, seed)
        self.samples: List[Tuple[str, int, int]] = splits[split]
        # 每项: (datasets_dir 下的相对路径, orientation_id, fill_id)

        self.transform = transform if transform is not None \
            else self._default_transform(split)

    @staticmethod
    def _read_csv(csv_path: str) -> List[dict]:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                compound = int(row["state_name"])
                if compound not in _COMPOUND_TO_DUAL:
                    continue
                ori_id, fill_id = _COMPOUND_TO_DUAL[compound]
                rel = row["image_path"].replace("\\", "/")
                rows.append({"rel": rel, "ori": ori_id, "fill": fill_id})
        return rows

    @staticmethod
    def _stratified_split(rows, ratios, seed):
        rng = random.Random(seed)
        by_combo: Dict[Tuple[int, int], list] = defaultdict(list)
        for r in rows:
            by_combo[(r["ori"], r["fill"])].append(r)

        splits = {"train": [], "val": [], "test": []}
        train_r, val_r = ratios[0], ratios[1]

        for combo_rows in by_combo.values():
            rng.shuffle(combo_rows)
            n = len(combo_rows)
            n_train = max(1, int(n * train_r))
            n_val   = max(0, int(n * val_r))
            splits["train"].extend(combo_rows[:n_train])
            splits["val"].extend(combo_rows[n_train: n_train + n_val])
            splits["test"].extend(combo_rows[n_train + n_val:])

        for key in splits:
            rng.shuffle(splits[key])
            splits[key] = [(r["rel"], r["ori"], r["fill"]) for r in splits[key]]

        print(
            f"[CupDataset] train={len(splits['train'])} | "
            f"val={len(splits['val'])} | test={len(splits['test'])}"
        )
        return splits

    @staticmethod
    def _default_transform(split: str):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        rel, ori_id, fill_id = self.samples[idx]
        img_path = os.path.join(self.datasets_dir, rel)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法加载图片 {img_path}: {e}")
        return self.transform(image), ori_id, fill_id

    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_counts  = torch.zeros(self.num_objects)
        fill_counts = torch.zeros(self.num_states)
        for _, ori_id, fill_id in self.samples:
            ori_counts[ori_id]   += 1
            fill_counts[fill_id] += 1

        def inv_freq(counts):
            w = 1.0 / (counts + 1e-6)
            return w / w.sum() * len(counts)

        return inv_freq(ori_counts), inv_freq(fill_counts)


def build_cup_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, CupDataset]:
    """
    Returns: train_loader, val_loader, test_loader, train_dataset
    接口与 build_dataloaders() 相同，train.py 无需其他改动。
    """
    dsets = {
        split: CupDataset(csv_path, split, split_ratio=split_ratio, seed=seed)
        for split in ("train", "val", "test")
    }

    def make_loader(split, shuffle):
        return DataLoader(
            dsets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return (
        make_loader("train", shuffle=True),
        make_loader("val",   shuffle=False),
        make_loader("test",  shuffle=False),
        dsets["train"],
    )