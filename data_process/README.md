**# 数据预处理模块（负责人：宋词）**



**## 文件说明**

**- `parse\_dataset.py` : 解析原始数据集，提取图片路径、物体名称、状态名称，生成中间文件。**

**- `split\_and\_encode.py` : 对标注进行分层划分（train/val/test），生成CSV和JSON映射文件。**

**- `augmentation\_demo.py` : 生成数据增强对比图，用于PPT展示。**

**- `processed/` : 存放中间文件 all\_annotations.csv（不提交到git，由脚本生成）。**



**## 使用方法**

**1. 先配置 `parse\_dataset.py` 中的路径和解析逻辑。**

**2. 运行 `python parse\_dataset.py`**

**3. 运行 `python split\_and\_encode.py`**

**4. 运行 `python augmentation\_demo.py` （可选）**



**## 当前进度**

**- 数据集下载中，待补充解析逻辑。**

