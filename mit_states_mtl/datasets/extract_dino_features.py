"""
用冻结的 DINOv2 提取 VAW 特征并缓存到磁盘
============================================
受限硬件下的最优策略：骨干冻结、特征只提取一次、训练只跑 MTL 头。
  - 复用 VAWDataset 的 bbox 裁剪 / resize 逻辑（augment=False）
  - DINOv2 ViT-S/14：输入 224×224 → CLS 特征 384 维
  - 特征存 fp16、属性标签存 int8（值 ∈ {1,0,-1}），省磁盘

用法:
    python datasets/extract_dino_features.py --ann-dir datasets/vaw_repo/data

输出（默认 datasets/vaw_features/）:
    {split}_feat.npy   [N, D]  float16
    {split}_obj.npy    [N]     int64
    {split}_attr.npy   [N, C]  int8   (1=正 0=负 -1=缺失)
    meta.json          obj2id / attr2id / 模型名 / 特征维度
    stats.npz          n_pos / n_neg（用于 RW-BCE 重加权）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# fmt: off
if sys.platform == "win32":
    import ctypes as _ct, importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _tlib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_tlib):
            os.add_dll_directory(_tlib)
            _iomp = os.path.join(_tlib, "libiomp5md.dll")
            if os.path.isfile(_iomp): _ct.CDLL(_iomp)
    _cbin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "Library", "bin"))
    if os.path.isdir(_cbin): os.add_dll_directory(_cbin)
# fmt: on

# 权重已缓存，离线加载，避免再依赖代理
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

VAW_ANN_DIR = "datasets/vaw_repo/data"
VG_IMG_DIRS = ["datasets/VG_images/VG_100K", "datasets/VG_images/VG_100K_2"]
OUT_DIR = "datasets/vaw_features"

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir",      default=VAW_ANN_DIR)
    p.add_argument("--img-dirs",     nargs="+", default=VG_IMG_DIRS)
    p.add_argument("--out-dir",      default=OUT_DIR)
    p.add_argument("--model",        default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--img-size",     type=int, default=224)
    p.add_argument("--batch",        type=int, default=128)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--min-obj-cnt",  type=int, default=30)
    p.add_argument("--min-attr-cnt", type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def extract_split(model, ds, device, mean, std, batch, workers, name):
    loader = DataLoader(ds, batch_size=batch, num_workers=workers,
                        shuffle=False, pin_memory=True)
    feats, objs, attrs = [], [], []
    t0 = time.time()
    n_total = len(loader)
    for i, (imgs, obj_id, attr_y) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True).float().div_(255)
        imgs = (imgs - mean) / std
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            f = model(imgs)                      # [B, D]
        feats.append(f.float().cpu().numpy().astype(np.float16))
        objs.append(obj_id.numpy().astype(np.int64))
        attrs.append(attr_y.numpy().astype(np.int8))
        if i == 0 or (i + 1) % 50 == 0:
            print(f"    [{name}] batch {i+1}/{n_total} | {time.time()-t0:.0f}s",
                  flush=True)
    return (np.concatenate(feats), np.concatenate(objs), np.concatenate(attrs))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, ".")
    from datasets.vaw_dataset import VAWDataset, build_vocab, compute_attr_stats

    print("构建词典...")
    obj2id, attr2id = build_vocab(
        os.path.join(args.ann_dir, "train.json"),
        args.min_obj_cnt, args.min_attr_cnt)
    print(f"  物体类别: {len(obj2id)}  属性类别: {len(attr2id)}")

    print("统计正负样本数（RW-BCE）...")
    n_pos, n_neg = compute_attr_stats(
        os.path.join(args.ann_dir, "train.json"), attr2id)

    # ── DINOv2 骨干（冻结）─────────────────────────────────────────
    import timm
    print(f"加载冻结骨干: {args.model} (img_size={args.img_size})")
    model = timm.create_model(args.model, pretrained=True, num_classes=0,
                              img_size=args.img_size)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    feat_dim = model.num_features
    print(f"  特征维度: {feat_dim}")

    mean = torch.tensor(IMG_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMG_STD,  device=device).view(1, 3, 1, 1)

    kw = dict(img_dirs=args.img_dirs, obj2id=obj2id, attr2id=attr2id,
              img_size=args.img_size, augment=False)

    for split in ["train", "val", "test"]:
        json_path = os.path.join(args.ann_dir, f"{split}.json")
        print(f"\n── 提取 {split} ──")
        ds = VAWDataset(json_path, **kw)
        print(f"  样本数: {len(ds)}")
        feat, obj, attr = extract_split(
            model, ds, device, mean, std, args.batch, args.workers, split)
        np.save(out / f"{split}_feat.npy", feat)
        np.save(out / f"{split}_obj.npy",  obj)
        np.save(out / f"{split}_attr.npy", attr)
        print(f"  已存: feat{feat.shape} obj{obj.shape} attr{attr.shape}")

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"obj2id": obj2id, "attr2id": attr2id,
                   "model": args.model, "feat_dim": int(feat_dim),
                   "img_size": args.img_size}, f, ensure_ascii=False, indent=2)
    np.savez(out / "stats.npz", n_pos=n_pos, n_neg=n_neg)
    print(f"\n完成。特征与元数据已写入 {out}/")


if __name__ == "__main__":
    main()