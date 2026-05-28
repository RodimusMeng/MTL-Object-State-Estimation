"""
对少数类（sideways/inverted）做数据增强扩充
=============================================
每张原图随机生成 N 张增强图，保存到 cup/<state>/，追加到 cup_dataset.csv。

用法:
    python datasets/augment_new_states.py              # 默认每类扩到 200 张
    python datasets/augment_new_states.py --target 300
    python datasets/augment_new_states.py --dry-run
"""

import argparse
import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DATASETS_DIR = Path(__file__).parent
CUP_DIR      = DATASETS_DIR / "cup"
CSV_PATH     = DATASETS_DIR / "cup_dataset.csv"

STATE_FOLDERS = {0: "empty", 1: "half", 2: "full", 3: "sideways", 4: "inverted"}
MINORITY_STATES = {3, 4}


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.3))
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, expand=True, fillcolor=(128, 128, 128))
    w, h = img.size
    scale = rng.uniform(0.80, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    left = rng.randint(0, w - nw)
    top  = rng.randint(0, h - nh)
    img  = img.crop((left, top, left + nw, top + nh))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return img


def load_existing() -> tuple[int, dict[int, list[Path]]]:
    max_idx = 0
    sources: dict[int, list[Path]] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"(\d+)\.\w+$", row["image_path"])
            if m:
                max_idx = max(max_idx, int(m.group(1)))
            sid = int(row["state_name"])
            if sid in MINORITY_STATES:
                rel  = row["image_path"].replace("\\", "/")
                absp = DATASETS_DIR / rel
                if absp.is_file():
                    sources.setdefault(sid, []).append(absp)
    return max_idx, sources


def run(target: int, seed: int, dry_run: bool):
    rng = random.Random(seed)

    print("[1] 读取现有数据...")
    max_idx, sources = load_existing()
    for sid, paths in sources.items():
        print(f"  state {sid} ({STATE_FOLDERS[sid]}): {len(paths)} 张原图")

    new_rows: list[dict] = []
    next_idx = max_idx + 1

    for sid, orig_paths in sources.items():
        current = len(orig_paths)
        need    = max(0, target - current)
        if need == 0:
            print(f"  state {sid}: 已有 {current} 张，无需扩充")
            continue
        print(f"\n[2] state {sid} ({STATE_FOLDERS[sid]}): {current} → {target}，生成 {need} 张...")
        state_dir = CUP_DIR / STATE_FOLDERS[sid]

        for i in range(need):
            src  = rng.choice(orig_paths)
            name = f"{next_idx:06d}.jpg"
            rel  = f"cup/{STATE_FOLDERS[sid]}/{name}"

            if not dry_run:
                img = Image.open(src).convert("RGB")
                img = augment(img, rng)
                img.save(state_dir / name, "JPEG", quality=92)

            new_rows.append({
                "image_path":  rel,
                "object_name": "cup",
                "state_name":  sid,
            })
            next_idx += 1

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{need}")

    if dry_run:
        print(f"\n[dry-run] 将生成 {len(new_rows)} 张，未实际写入。")
        return

    if not new_rows:
        print("\n无需生成，退出。")
        return

    print(f"\n[3] 追加 {len(new_rows)} 行到 CSV...")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "object_name", "state_name"]
        )
        writer.writerows(new_rows)

    print("[4] 重新生成 YOLO 数据集...")
    import subprocess, sys
    subprocess.run(
        [sys.executable, str(DATASETS_DIR / "convert_to_yolo.py")],
        cwd=DATASETS_DIR.parent,
    )

    print("\n完成! 最终分布:")
    from collections import Counter
    counts = Counter()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[int(row["state_name"])] += 1
    for sid, cnt in sorted(counts.items()):
        print(f"  {sid} ({STATE_FOLDERS[sid]:8s}): {cnt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target",  type=int, default=200)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(target=args.target, seed=args.seed, dry_run=args.dry_run)