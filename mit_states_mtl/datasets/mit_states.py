"""
MIT States Dataset Loader for Multi-Task Learning
==================================================
MIT States dataset 目录结构:
    <root>/
        images/
            <adj> <noun>/      ← 文件夹名即 (state, object) 标签，空格分隔
                *.jpg
            adj <noun>/        ← "adj" 是特殊标签，表示无特定状态修饰的原始图片

    例: "broken bottle/"、"wet dog/"、"adj apple/"

本模块从文件夹名自动解析标签，无需依赖外部 metadata 文件。
支持分层采样 (stratified) train/val/test split，并提供数据增强流水线。
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  核心 Dataset 类
# ─────────────────────────────────────────────────────────────────────────────

class MITStatesDataset(Dataset):
    """
    MIT States 多任务分类数据集。

    每个样本返回:
        image    : FloatTensor [3, H, W]
        obj_id   : int — 物体类别 ID (0 ~ num_objects-1)
        state_id : int — 状态类别 ID (0 ~ num_states-1)

    Args:
        root        : release_dataset/ 根目录（其下含 images/ 子目录）
        split       : "train" | "val" | "test"
        transform   : torchvision transform，None 时使用默认
        split_ratio : (train, val, test) 比例，默认 (0.70, 0.15, 0.15)
        seed        : 随机种子，保证 split 可复现
        min_samples : 每个 (state, object) 组合至少需要的样本数，
                      低于此值的组合被过滤（清洗长尾噪声）
        cache_split : 将 split 结果缓存为 JSON，避免重复计算
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        min_samples: int = 2,
        cache_split: bool = True,
    ):
        assert split in ("train", "val", "test"), \
            f"split 必须是 train/val/test，收到 '{split}'"
        assert abs(sum(split_ratio) - 1.0) < 1e-6, \
            "split_ratio 三项之和必须为 1.0"

        self.root = root
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.min_samples = min_samples

        # ── 解析标签词汇表 ─────────────────────────────────────────
        self.obj2id, self.state2id = self._build_vocab()
        self.id2obj = {v: k for k, v in self.obj2id.items()}
        self.id2state = {v: k for k, v in self.state2id.items()}

        # ── 加载 / 计算 split ──────────────────────────────────────
        cache_path = os.path.join(root, f"_split_cache_s{seed}.json")
        if cache_split and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                all_splits = json.load(f)
        else:
            all_splits = self._stratified_split()
            if cache_split:
                with open(cache_path, "w") as f:
                    json.dump(all_splits, f)

        self.samples: List[Tuple[str, int, int]] = all_splits[split]
        # 每项: [image_path, obj_id, state_id]

        # ── 数据增强流水线 ─────────────────────────────────────────
        self.transform = transform if transform is not None \
            else self._default_transform(split)

    # ──────────────────────────────────────────────────────────────
    #  标签词汇表构建
    # ──────────────────────────────────────────────────────────────

    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        从 images/ 子目录名解析所有 (state, object) 对。
        目录格式: "<adj> <noun>"（空格分隔），例如 "broken bottle"。
        "adj <noun>" 是特殊目录，state="adj" 表示无特定修饰的原始图片。
        """
        img_dir = os.path.join(self.root, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"找不到 images 目录: {img_dir}\n"
                "请确认 root 指向 release_dataset/ 目录，\n"
                "即 images/ 子目录应在 <root>/images/ 下。"
            )

        objects = set()
        states = set()
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            if not os.path.isdir(folder_path):
                continue  # 跳过 .DS_Store 等非目录文件
            parts = folder.split(" ", 1)  # 空格分隔，maxsplit=1
            if len(parts) != 2:
                continue
            state, obj = parts
            states.add(state)
            objects.add(obj)

        obj2id   = {o: i for i, o in enumerate(sorted(objects))}
        state2id = {s: i for i, s in enumerate(sorted(states))}
        return obj2id, state2id

    # ──────────────────────────────────────────────────────────────
    #  分层采样 Split
    # ──────────────────────────────────────────────────────────────

    def _stratified_split(self) -> Dict[str, List]:
        """
        对每个 (state, object) 组合分别做 train/val/test 切分，
        保证各 split 中类别分布一致（分层采样）。
        过滤掉样本数不足 min_samples 的组合。
        """
        img_dir = os.path.join(self.root, "images")
        rng = random.Random(self.seed)

        combo_to_paths: Dict[Tuple[int, int], List[str]] = defaultdict(list)

        for folder in sorted(os.listdir(img_dir)):
            folder_path = os.path.join(img_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split(" ", 1)  # 空格分隔
            if len(parts) != 2:
                continue
            state, obj = parts
            if state not in self.state2id or obj not in self.obj2id:
                continue

            state_id = self.state2id[state]
            obj_id   = self.obj2id[obj]

            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    # 存绝对路径：避免 DataLoader worker 子进程 CWD 不一致问题
                    abs_path = os.path.abspath(os.path.join(folder_path, fname))
                    if os.path.isfile(abs_path):   # 过滤真实缺失的文件
                        combo_to_paths[(obj_id, state_id)].append(abs_path)

        train_r, val_r, _ = self.split_ratio
        splits: Dict[str, List] = {"train": [], "val": [], "test": []}

        for (obj_id, state_id), paths in combo_to_paths.items():
            if len(paths) < self.min_samples:
                continue  # 长尾过滤

            rng.shuffle(paths)
            n = len(paths)
            n_train = max(1, int(n * train_r))
            n_val   = max(0, int(n * val_r))

            train_p = paths[:n_train]
            val_p   = paths[n_train: n_train + n_val]
            test_p  = paths[n_train + n_val:]

            for p in train_p:
                splits["train"].append([p, obj_id, state_id])
            for p in val_p:
                splits["val"].append([p, obj_id, state_id])
            for p in test_p:
                splits["test"].append([p, obj_id, state_id])

        for key in splits:
            rng.shuffle(splits[key])

        print(
            f"[MITStates] Split 完成 | "
            f"train={len(splits['train'])} | "
            f"val={len(splits['val'])} | "
            f"test={len(splits['test'])} | "
            f"#objects={len(self.obj2id)} | "
            f"#states={len(self.state2id)}"
        )
        return splits

    # ──────────────────────────────────────────────────────────────
    #  数据增强
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _default_transform(split: str) -> transforms.Compose:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]

        if split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ])

    # ──────────────────────────────────────────────────────────────
    #  Dataset 接口
    # ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path, obj_id, state_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法加载图片 {img_path}: {e}")
        image = self.transform(image)
        return image, obj_id, state_id

    # ──────────────────────────────────────────────────────────────
    #  辅助属性
    # ──────────────────────────────────────────────────────────────

    @property
    def num_objects(self) -> int:
        return len(self.obj2id)

    @property
    def num_states(self) -> int:
        return len(self.state2id)

    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 object 和 state 的逆频率类别权重，用于处理长尾分布。
        返回 (obj_weights, state_weights)，可直接传给 CrossEntropyLoss(weight=...)
        """
        obj_counts   = torch.zeros(self.num_objects)
        state_counts = torch.zeros(self.num_states)
        for _, obj_id, state_id in self.samples:
            obj_counts[obj_id]     += 1
            state_counts[state_id] += 1

        obj_weights   = 1.0 / (obj_counts   + 1e-6)
        obj_weights   = obj_weights   / obj_weights.sum()   * self.num_objects

        state_weights = 1.0 / (state_counts + 1e-6)
        state_weights = state_weights / state_weights.sum() * self.num_states

        return obj_weights, state_weights


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    min_samples: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, "MITStatesDataset"]:
    """
    一次性构建 train/val/test 三个 DataLoader。

    Returns:
        train_loader, val_loader, test_loader, train_dataset
    """
    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = MITStatesDataset(
            root=root,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            min_samples=min_samples,
        )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, datasets["train"]
