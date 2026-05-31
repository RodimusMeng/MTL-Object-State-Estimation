"""
OSDD 数据集配置脚本
===================
从 https://github.com/philipposg/OSDD 下载数据后运行本脚本。

预期目录结构（下载解压后）:
    datasets/osdd/
        TRAIN/
            images/  (或 obj_train_data/)
            valid_paths.txt (或 train.txt)
        VAL/
            images/
            valid_paths.txt
        TEST/
            images/
        obj.names   (类别名称)

运行:
    python datasets/setup_osdd.py --osdd-dir datasets/osdd
"""

import argparse
import os
import shutil
from pathlib import Path


# OSDD 原始 9 个状态类别
OSDD_CLASSES = [
    "open",
    "close",
    "empty",
    "containing_liquid",
    "containing_solid",
    "plugged",
    "unplugged",
    "folded",
    "unfolded",
]


def find_images(directory: Path):
    """递归找出所有图片文件"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def find_labels(directory: Path):
    """递归找出所有标注文件"""
    return [p for p in directory.rglob("*.txt")
            if p.name not in ("valid_paths.txt", "train.txt", "val.txt", "test.txt", "obj.names")]


def write_yaml(out_path: Path, nc: int, names: list, data_root: Path):
    content = (
        f"path: {data_root.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\nnc: {nc}\n"
        f"names:\n"
    )
    for name in names:
        content += f"  - {name}\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"YAML 写入: {out_path}")


def reorganize(osdd_dir: Path):
    """
    将 OSDD 的 TRAIN/VAL/TEST 结构重组为 YOLO 标准结构:
        images/train/, images/val/, images/test/
        labels/train/, labels/val/, labels/test/
    """
    split_map = {
        "TRAIN": "train",
        "TRAIN_DATA": "train",
        "VAL": "val",
        "VALIDATION": "val",
        "TEST": "test",
    }

    for src_name, dst_name in split_map.items():
        src = osdd_dir / src_name
        if not src.is_dir():
            continue

        imgs_out = osdd_dir / "images" / dst_name
        lbls_out = osdd_dir / "labels" / dst_name
        imgs_out.mkdir(parents=True, exist_ok=True)
        lbls_out.mkdir(parents=True, exist_ok=True)

        imgs = find_images(src)
        lbls = find_labels(src)

        for img in imgs:
            dst = imgs_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

        for lbl in lbls:
            dst = lbls_out / lbl.name
            if not dst.exists():
                shutil.copy2(lbl, dst)

        print(f"  {src_name} → {dst_name}: {len(imgs)} 图片, {len(lbls)} 标注")


def check_names_file(osdd_dir: Path) -> list:
    """查找类别名称文件"""
    candidates = list(osdd_dir.rglob("obj.names")) + list(osdd_dir.rglob("*.names"))
    if candidates:
        names = [l.strip() for l in candidates[0].read_text().splitlines() if l.strip()]
        print(f"从 {candidates[0].name} 读取类别: {names}")
        return names
    print(f"未找到 .names 文件，使用默认 OSDD 类别: {OSDD_CLASSES}")
    return OSDD_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--osdd-dir", default="datasets/osdd", help="OSDD 解压目录")
    args = p.parse_args()

    osdd_dir = Path(args.osdd_dir)
    if not osdd_dir.is_dir():
        print(f"目录不存在: {osdd_dir}")
        print("请先从 https://github.com/philipposg/OSDD 下载并解压数据到该目录")
        return

    print(f"处理 OSDD 数据: {osdd_dir}")
    names = check_names_file(osdd_dir)
    reorganize(osdd_dir)

    yaml_path = osdd_dir / "osdd.yaml"
    write_yaml(yaml_path, len(names), names, osdd_dir)

    print(f"\n完成！训练命令（Exp D）:")
    print(f"  python train_yolo.py --model yolov8s.pt --data {yaml_path} --name osdd_yolov8s --epochs 100")
    print(f"\nExp E1 (轻量):")
    print(f"  python train_yolo.py --model yolov8n.pt --data {yaml_path} --name osdd_yolov8n --epochs 100")
    print(f"\nExp E2 (大模型):")
    print(f"  python train_yolo.py --model yolov8m.pt --data {yaml_path} --name osdd_yolov8m --epochs 100")
    print(f"\nExp F (无预训练):")
    print(f"  python train_yolo.py --model yolov8s.yaml --data {yaml_path} --name osdd_no_pretrain --epochs 100")


if __name__ == "__main__":
    main()
