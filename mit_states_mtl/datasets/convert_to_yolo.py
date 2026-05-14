"""
将 cup_dataset.csv 转换为 YOLOv8 检测格式
==========================================
输出目录结构:
    cup_yolo/
    ├── images/train/  images/val/  images/test/
    ├── labels/train/  labels/val/  labels/test/
    └── cup_yolo.yaml

标签格式（每行一个目标）:
    <class_id> <x_center> <y_center> <width> <height>   (归一化 0~1)

当前图片为已裁剪杯子图，bbox 设为全图 (0.5 0.5 1.0 1.0)。
class_id 直接使用 CSV 中的 state_name 数字，对应复合类:
    0=cup_empty  1=cup_half  2=cup_full  3=cup_sideways  4=cup_inverted

用法:
    python datasets/convert_to_yolo.py
    python datasets/convert_to_yolo.py --csv datasets/cup_dataset.csv --out datasets/cup_yolo --seed 42
"""

import argparse
import csv
import os
import random
import shutil


CLASSES = ["cup_empty", "cup_half", "cup_full", "cup_sideways", "cup_inverted"]

# 全图伪 bbox (x_center, y_center, w, h) 归一化
FULL_IMAGE_BBOX = "0.5 0.5 1.0 1.0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="datasets/cup_dataset.csv")
    p.add_argument("--img_root", default="datasets",
                   help="cup_selected/ 所在的父目录")
    p.add_argument("--out", default="datasets/cup_yolo")
    p.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "img": row["image_path"].replace("\\", "/"),
                "state": int(row["state_name"]),
            })
    return rows


def split_data(rows: list[dict], ratios: list[float], seed: int):
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


def write_yolo(splits: dict, img_root: str, out_dir: str):
    for split_name, rows in splits.items():
        img_out = os.path.join(out_dir, "images", split_name)
        lbl_out = os.path.join(out_dir, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for row in rows:
            src = os.path.join(img_root, row["img"])
            if not os.path.isfile(src):
                print(f"  [skip] 找不到图片: {src}")
                continue

            fname = os.path.basename(row["img"])
            stem  = os.path.splitext(fname)[0]

            shutil.copy2(src, os.path.join(img_out, fname))

            label_line = f"{row['state']} {FULL_IMAGE_BBOX}\n"
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                f.write(label_line)

        print(f"  {split_name:5s}: {len(rows):4d} 张")


def write_yaml(out_dir: str):
    abs_out = os.path.abspath(out_dir)
    yaml_content = f"""\
path: {abs_out}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = os.path.join(out_dir, "cup_yolo.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  yaml: {yaml_path}")


def main():
    args = parse_args()
    assert abs(sum(args.split) - 1.0) < 1e-6, "split 比例之和必须为 1.0"

    print("[1] 读取 CSV...")
    rows = read_csv(args.csv)
    print(f"    共 {len(rows)} 条记录")

    print("[2] 分层划分 train/val/test...")
    splits = split_data(rows, args.split, args.seed)

    print("[3] 写出 YOLO 格式文件...")
    write_yolo(splits, args.img_root, args.out)

    print("[4] 生成 cup_yolo.yaml...")
    write_yaml(args.out)

    print("\n完成! 目录:", args.out)
    print("类别映射:")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
