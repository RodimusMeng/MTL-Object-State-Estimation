"""
将 cup_dataset.csv 转换为 YOLOv8 检测格式
==========================================
输出目录结构:
    cup/yolo/
    ├── images/train/  images/val/  images/test/
    ├── labels/train/  labels/val/  labels/test/
    └── cup.yaml

标签格式（每行一个目标）:
    <class_id> <x_center> <y_center> <width> <height>   (归一化 0~1)

当前图片为已裁剪杯子图，bbox 设为全图 (0.5 0.5 1.0 1.0)。
class_id 直接使用 CSV 中的 state_name 数字:
    0=cup_empty  1=cup_half  2=cup_full  3=cup_sideways  4=cup_inverted

用法:
    python datasets/convert_to_yolo.py
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


CLASSES = ["cup_empty", "cup_half", "cup_full", "cup_sideways", "cup_inverted"]
FULL_IMAGE_BBOX = "0.5 0.5 1.0 1.0"

DATASETS_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default=str(DATASETS_DIR / "cup_dataset.csv"))
    p.add_argument("--out",  default=str(DATASETS_DIR / "cup" / "yolo"))
    p.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "img": row["image_path"].replace("\\", "/"),
                "state": int(row["state_name"]),
            })
    return rows


def split_data(rows, ratios, seed):
    rng = random.Random(seed)
    by_state: dict[int, list] = {}
    for row in rows:
        by_state.setdefault(row["state"], []).append(row)

    splits = {"train": [], "val": [], "test": []}
    train_r, val_r = ratios[0], ratios[1]

    for state_rows in by_state.values():
        rng.shuffle(state_rows)
        n = len(state_rows)
        n_train = max(1, int(n * train_r))
        n_val   = max(0, int(n * val_r))
        splits["train"].extend(state_rows[:n_train])
        splits["val"].extend(state_rows[n_train: n_train + n_val])
        splits["test"].extend(state_rows[n_train + n_val:])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def write_yolo(splits, out_dir: str):
    out = Path(out_dir)
    for split_name, rows in splits.items():
        img_out = out / "images" / split_name
        lbl_out = out / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for row in rows:
            src = DATASETS_DIR / row["img"]
            if not src.is_file():
                print(f"  [skip] {src}")
                continue
            fname = src.name
            shutil.copy2(src, img_out / fname)
            (lbl_out / (src.stem + ".txt")).write_text(
                f"{row['state']} {FULL_IMAGE_BBOX}\n"
            )

        print(f"  {split_name:5s}: {len(rows):4d} 张")


def write_yaml(out_dir: str):
    out = Path(out_dir)
    yaml_content = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (out / "cup.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"  yaml: {out / 'cup.yaml'}")


def main():
    args = parse_args()
    assert abs(sum(args.split) - 1.0) < 1e-6

    print("[1] 读取 CSV...")
    rows = read_csv(args.csv)
    print(f"    共 {len(rows)} 条记录")

    print("[2] 分层划分 train/val/test...")
    splits = split_data(rows, args.split, args.seed)

    print("[3] 写出 YOLO 格式文件...")
    write_yolo(splits, args.out)

    print("[4] 生成 cup.yaml...")
    write_yaml(args.out)

    print(f"\n完成! 目录: {args.out}")


if __name__ == "__main__":
    main()