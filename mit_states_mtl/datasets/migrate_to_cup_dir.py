"""
一次性迁移脚本：将 cup_selected/ + cup_other/ 整合为 cup/<state>/ 结构
=========================================================================
执行后的结构:
    datasets/cup/
        empty/      ← state 0
        half/       ← state 1
        full/       ← state 2
        sideways/   ← state 3
        inverted/   ← state 4

cup_dataset.csv 中的路径同步更新为 cup/<state>/filename 格式。

用法:
    python datasets/migrate_to_cup_dir.py             # 正式执行
    python datasets/migrate_to_cup_dir.py --dry-run   # 预览
"""

import argparse
import csv
import shutil
from pathlib import Path

DATASETS_DIR = Path(__file__).parent

STATE_FOLDERS = {0: "empty", 1: "half", 2: "full", 3: "sideways", 4: "inverted"}

CSV_PATH     = DATASETS_DIR / "cup_dataset.csv"
CUP_DIR      = DATASETS_DIR / "cup"
OLD_SELECTED = DATASETS_DIR / "cup_selected"
OLD_OTHER    = DATASETS_DIR / "cup_other"


def run(dry_run: bool):
    # ── 读取 CSV ────────────────────────────────────────────────────
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"CSV 共 {len(rows)} 行")

    # ── 建目标目录 ──────────────────────────────────────────────────
    for folder in STATE_FOLDERS.values():
        (CUP_DIR / folder).mkdir(parents=True, exist_ok=True)
    (CUP_DIR / "yolo").mkdir(exist_ok=True)
    print(f"目标目录: {CUP_DIR}")

    # ── 迁移图片 ────────────────────────────────────────────────────
    new_rows = []
    missing  = 0

    for row in rows:
        rel = row["image_path"].replace("\\", "/")
        state_id = int(row["state_name"])
        folder   = STATE_FOLDERS[state_id]
        fname    = Path(rel).name

        src = DATASETS_DIR / rel
        dst = CUP_DIR / folder / fname
        new_rel = f"cup/{folder}/{fname}"

        if not src.is_file():
            print(f"  [缺失] {src}")
            missing += 1
            continue

        if not dry_run:
            shutil.copy2(src, dst)

        new_rows.append({
            "image_path":  new_rel,
            "object_name": row["object_name"],
            "state_name":  state_id,
        })

    print(f"迁移: {len(new_rows)} 张  缺失: {missing} 张")

    # ── 写新 CSV ────────────────────────────────────────────────────
    if not dry_run:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_path", "object_name", "state_name"]
            )
            writer.writeheader()
            writer.writerows(new_rows)
        print("CSV 已更新")

    # ── 汇总 ────────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(r["state_name"] for r in new_rows)
    for sid, cnt in sorted(dist.items()):
        print(f"  cup/{STATE_FOLDERS[int(sid)]}/: {cnt} 张")

    if dry_run:
        print("\n[dry-run] 未实际写入。")
    else:
        print("\n迁移完成。请手动删除旧目录:")
        print(f"  {OLD_SELECTED}")
        print(f"  {OLD_OTHER}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(dry_run=args.dry_run)