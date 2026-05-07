"""
MIT States 数据集下载说明脚本
=============================
MIT States 官方地址:
    http://web.mit.edu/phillipi/Public/states_and_transformations/index.html

数据集约 450MB（压缩），解压后约 1.7GB。
目录结构: images/<adj> <noun>/  （空格分隔，如 "broken bottle"）

用法:
    python download_data.py --dest ./datasets/release_dataset
    python download_data.py --skip-download   # 只打印说明
"""

import argparse
import os
import tarfile
import zipfile
import urllib.request

MIT_STATES_URL = (
    "http://wednesday.csail.mit.edu/joseph_result/"
    "state_and_transformation/release_dataset.zip"
)


def download_with_progress(url: str, dest_path: str):
    print(f"下载: {url}")

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f"\r  {pct:5.1f}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB",
                  end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, hook)
    print()


def extract_archive(archive_path: str, dest_dir: str):
    print(f"解压 {archive_path} → {dest_dir}")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    print("解压完成。")


def check_structure(root: str) -> bool:
    img_dir = os.path.join(root, "images")
    if not os.path.isdir(img_dir):
        return False
    folders = [f for f in os.listdir(img_dir)
               if os.path.isdir(os.path.join(img_dir, f)) and " " in f]
    return len(folders) > 100


def main(dest: str, skip_download: bool):
    os.makedirs(dest, exist_ok=True)

    if check_structure(dest):
        print(f"[download_data] 数据集已存在于 {dest}，跳过下载。")
        return

    if skip_download:
        print(
            "[download_data] 请手动下载并解压数据集：\n"
            f"  URL: {MIT_STATES_URL}\n"
            f"  解压到: {dest}/\n"
            "  解压后应有: images/<adj> <noun>/ 子目录（空格分隔）"
        )
        return

    archive = os.path.join(dest, "release_dataset.zip")
    if not os.path.exists(archive):
        download_with_progress(MIT_STATES_URL, archive)
    extract_archive(archive, dest)

    if check_structure(dest):
        print(f"[download_data] 数据集就绪: {dest}")
    else:
        print("[download_data] 警告：目录结构异常，请检查 images/ 子目录。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest",          default="./datasets/release_dataset")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()
    main(args.dest, args.skip_download)
