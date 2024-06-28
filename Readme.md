江南大学 花政林 Locally Weighted Ensemble Clustering复现：

---

# 局部加权集成聚类论文复现

## 介绍
此代码库基于Python实现了“局部加权集成聚类”（LWEA和LWGP）的算法，参考了原Matlab开源代码。此复现主要目的是深入理解原论文提出的方法，并提升编程及数据分析能力。

## 文件描述
- `veiw_dataset.py`: 加载和显示数据集的内容。主要函数load_and_view_data接受一个文件路径参数，加载.mat文件，并打印文件中的所有变量名、数据类型以及标签范围。
- `compute_ECI.py`: 计算熵加权集成指数（ECI）。包含函数compute_eci，它通过熵的概念评估各个聚类的质量和可靠性。
- `compute_LWCA.py`: 实现计算局部加权共同关联矩阵（LWCA）的功能。
- `compute_NMI.py`: 实现计算标准化互信息（NMI）的功能，用于评估聚类效果。
- `compute_ARI.py`: 实现计算调整兰德指数（ARI）的功能，用于评估聚类效果。
- `run_LWEA.py` 和 `run_LWGP.py`: 分别用于运行LWEA和LWGP聚类算法。
- `sparse_matrix_generator.py`: 生成稀疏矩阵并统一聚类标签。
- `LWEA_and_LWGP.py`: 综合以上脚本，提供完整的实验流程。
- `plot_NMI_and_ARI_fig.py`: 绘制不同集成大小M下的基于各数据集的该方法平均性能。
- `Debug.txt`: 复现过程中遇到的问题或调试过程。

## 环境需求
- Python 3.8 或更高版本
- NumPy
- SciPy
- scikit-learn

## 运行指南
在Windows环境下运行以下命令来执行LWEA与LWGP算法并评估其性能：
```bash
python LWEA_and_LWGP.py
```

## 数据集说明
此代码使用.mat格式的数据文件，其包含基础聚类结果（members）和真实聚类标签（gt）两部分。数据文件存放于`./Dataset/`目录下。

## 引用
此代码方法来源于以下论文：
```
Dong Huang, Chang-Dong Wang, Jian-Huang Lai, "Locally Weighted Ensemble Clustering," IEEE Transactions on Cybernetics, 2018.
```
