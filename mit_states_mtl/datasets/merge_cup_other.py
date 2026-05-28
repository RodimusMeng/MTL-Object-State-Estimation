"""
将 cup_other/ 里的新图片整合进主数据集
========================================
- 自动处理 iPhone EXIF 旋转
- 复制到 cup/<state>/ 目录，续接编号
- 追加到 cup_dataset.csv
- 完成后自动重新生成 YOLO 格式数据集

cup_other/ 目录结构（子目录名即状态名）:
    cup_other/sideways/   → state 3
    cup_other/inverted/   → state 4

新增状态时，在此添加映射并在 cup/ 下建对应目录即可:
    FOLDER_TO_STATE = { "sideways": 3, "inverted": 4, "新状态": 5 }

用法:
    python datasets/merge_cup_other.py
    python datasets/merge_cup_other.py --dry-run
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps


DATASETS_DIR = Path(__file__).parent
CUP_DIR      = DATASETS_DIR / "cup"
CSV_PATH     = DATASETS_DIR / "cup_dataset.csv"
CUP_OTHER    = DATASETS_DIR / "cup_other"

# 子目录名 → state_id 映射（新增状态时在此追加）
FOLDER_TO_STATE = {
    "sideways": 3,
    "inverted": 4,
}
STATE_NAMES = {v: k for k, v in FOLDER_TO_STATE.items()}


def last_index(csv_path: Path) -> int:
    max_idx = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.\w+$", row["image_path"])
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def collect_sources() -> list[tuple[Path, int]]:
    sources = []
    for folder_name, state_id in FOLDER_TO_STATE.items():
        folder = CUP_OTHER / folder_name
        if not folder.is_dir():
            print(f"  [警告] 找不到: {folder}")
            continue
        imgs = sorted(
            p for p in folder.iterdir()
            if p.suffix.upper() in (".JPG", ".JPEG", ".PNG", ".WEBP")
        )
        print(f"  {folder_name} (state={state_id}): {len(imgs)} 张")
        sources.extend((p, state_id) for p in imgs)
    return sources


def run(dry_run: bool):
    print("[1] 扫描 cup_other/ ...")
    sources = collect_sources()
    if not sources:
        print("没有找到图片，退出。")
        return

    print(f"\n[2] 续接编号...")
    start_idx = last_index(CSV_PATH) + 1
    print(f"    从 {start_idx:06d} 开始")

    new_rows = []
    for i, (src_path, state_id) in enumerate(sources):
        idx      = start_idx + i
        dst_name = f"{idx:06d}.jpg"
        state    = STATE_NAMES[state_id]
        dst_path = CUP_DIR / state / dst_name
        rel_path = f"cup/{state}/{dst_name}"

        print(f"  {'[dry]' if dry_run else '[copy]'} "
              f"{src_path.name} → {rel_path}")

        if not dry_run:
            img = Image.open(src_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img.save(dst_path, "JPEG", quality=95)

        new_rows.append({
            "image_path":  rel_path,
            "object_name": "cup",
            "state_name":  state_id,
        })

    if dry_run:
        print(f"\n[dry-run] 将追加 {len(new_rows)} 行，未实际写入。")
        return

    print(f"\n[3] 追加 {len(new_rows)} 行到 CSV...")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "object_name", "state_name"]
        )
        writer.writerows(new_rows)

    print("[4] 重新生成 YOLO 数据集...")
    subprocess.run(
        [sys.executable, str(DATASETS_DIR / "convert_to_yolo.py")],
        cwd=DATASETS_DIR.parent,
    )
    print("完成!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(dry_run=args.dry_run)