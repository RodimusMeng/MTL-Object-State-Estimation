import os
import pandas as pd

# ======= 配置区域（稍后根据实际数据集修改） ========
DATASET_ROOT = "../data/raw_datasets/your_dataset_name"  # 待修改
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
ANNOTATION_FILE = os.path.join(DATASET_ROOT, "annotations.csv")  # 待修改

# ======= 解析函数（根据实际标注格式修改） ========
def parse_annotations():
    # TODO: 根据你的数据集格式实现
    # 这里先返回一个空 DataFrame 作为占位
    # 后续我们会替换成真实代码
    df = pd.DataFrame(columns=["image_path", "object_name", "state_name"])
    return df

# ======= 主流程 ========
if __name__ == "__main__":
    df = parse_annotations()
    os.makedirs("processed", exist_ok=True)
    df.to_csv("processed/all_annotations.csv", index=False)
    print(f"已解析 {len(df)} 条标注，保存至 processed/all_annotations.csv")