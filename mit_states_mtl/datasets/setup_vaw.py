"""
VAW 数据集配置脚本
==================
运行此脚本验证标注和图片是否就绪。

用法:
    python datasets/setup_vaw.py
    python datasets/setup_vaw.py --ann-dir datasets/vaw_annotations/data
                                 --img-dirs datasets/VG_images/VG_100K
                                            datasets/VG_images/VG_100K_2
"""

import argparse
import json
import os
import sys
from pathlib import Path

ANN_DIR  = "datasets/vaw_annotations/data"
IMG_DIRS = [
    "datasets/VG_images/VG_100K",
    "datasets/VG_images/VG_100K_2",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir",  default=ANN_DIR)
    p.add_argument("--img-dirs", nargs="+", default=IMG_DIRS)
    return p.parse_args()


def check_annotations(ann_dir: str):
    print("\n── 标注文件检查 ─────────────────────────────────")
    required = ["train.json", "val.json", "test.json"]
    all_ok = True
    for fname in required:
        p = os.path.join(ann_dir, fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  ✓ {fname}  ({len(data):,} 条)")
        else:
            print(f"  ✗ {fname}  未找到 → {p}")
            all_ok = False
    return all_ok


def check_images(ann_dir: str, img_dirs: list, sample_n: int = 200):
    print("\n── 图片目录检查 ─────────────────────────────────")
    total_imgs = 0
    for d in img_dirs:
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith(".jpg")])
            print(f"  ✓ {d}  ({n:,} 张)")
            total_imgs += n
        else:
            print(f"  ✗ {d}  目录不存在")
    print(f"  合计: {total_imgs:,} 张")

    # 抽查 sample_n 条标注能否找到对应图片
    train_json = os.path.join(ann_dir, "train.json")
    if not os.path.exists(train_json):
        return

    with open(train_json, encoding="utf-8") as f:
        data = json.load(f)

    import random
    sample = random.sample(data, min(sample_n, len(data)))
    found = missing = 0
    missing_ids = []
    for d in sample:
        iid  = d["image_id"]
        fname = f"{iid}.jpg"
        hit = any(os.path.exists(os.path.join(dd, fname)) for dd in img_dirs)
        if hit:
            found += 1
        else:
            missing += 1
            missing_ids.append(iid)

    coverage = found / (found + missing) if (found + missing) > 0 else 0
    print(f"\n  抽查 {sample_n} 条标注：找到 {found}，缺失 {missing}（覆盖率 {coverage:.1%}）")
    if missing_ids[:5]:
        print(f"  缺失示例 image_id: {missing_ids[:5]}")

    if coverage < 0.9:
        print("\n  ⚠️  图片覆盖率低于 90%，请确认 VG_100K + VG_100K_2 均已解压到对应目录")
    else:
        print("\n  ✓ 图片覆盖率正常，可以开始训练")


def show_sample(ann_dir: str, img_dirs: list):
    print("\n── 样本预览（前 3 条）──────────────────────────")
    train_json = os.path.join(ann_dir, "train.json")
    if not os.path.exists(train_json):
        return
    with open(train_json, encoding="utf-8") as f:
        data = json.load(f)
    for d in data[:3]:
        print(f"  image_id={d['image_id']}  object={d['object_name']}")
        print(f"    pos_attrs: {d.get('positive_attributes', [])[:5]}")
        print(f"    neg_attrs: {d.get('negative_attributes', [])[:3]}")


def main():
    args = parse_args()
    ann_ok = check_annotations(args.ann_dir)
    check_images(args.ann_dir, args.img_dirs)
    show_sample(args.ann_dir, args.img_dirs)

    print("\n────────────────────────────────────────────────")
    if ann_ok:
        print("标注就绪，图片下载完成后运行:")
        print("  python train_vaw_mtl.py")
    else:
        print("请先下载 VAW 标注:")
        print("  pip install huggingface_hub")
        print("  python -c \"from huggingface_hub import snapshot_download;")
        print("    snapshot_download('mikewang/vaw', repo_type='dataset',")
        print("    local_dir='datasets/vaw_annotations')\"")
        print()
        print("或从 GitHub 下载 data/ 目录:")
        print("  https://github.com/adobe-research/vaw_dataset/tree/main/data")


if __name__ == "__main__":
    main()
