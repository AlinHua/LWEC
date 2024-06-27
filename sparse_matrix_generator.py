import numpy as np
import scipy.sparse as sp


def generate_sparse_matrix(base_cls):
    # 获取基础聚类的尺寸
    N, M = base_cls.shape  # N: 数据点数量, M: 基础聚类的数量

    # 初始化基础聚类
    bcs = base_cls.copy()
    # 统一聚类标签(每个基础聚类的标签将从之前的开始，确保在全局范围内唯一)
    n_cls_orig = np.max(bcs, axis=0)  # 每行最大值
    C = np.hstack(([0], np.cumsum(n_cls_orig)[:-1]))  # 最大值累加和并偏移
    bcs += C.astype(np.uint8)  # 每行加上最大值累加和
    # 计算全局聚类数
    n_cls = np.max(bcs) - np.min(bcs) + 1

    # 生成稀疏矩阵(CSR格式存储: 使用三个一维数组分别存储非零元素的值、列索引和行偏移量)
    # 行为类别，列为数据点
    row = bcs.flatten() - np.min(bcs)  # 展开 bcs 矩阵为1D向量(标签不一定从0开始), 生成聚类的索引
    col = np.repeat(np.arange(N), M)  # 生成数据点索引
    data = np.ones(N * M)  # 创建一个全为1的数组，用于填充稀疏矩阵，表示对应位置的数据点属于某个聚类
    # 创建稀疏矩阵 (n_cls, N): 二值矩阵，用于表示哪些数据点属于哪些聚类baseClsSegs[i, j] = 1，则表示数据点 j 属于聚类 i + 1
    base_cls_segs = sp.csr_matrix((data, (row, col)), shape=(n_cls, N))  # 对于没有在 row 和 col 中指定的位置，矩阵会自动填充 0

    return bcs, base_cls_segs  # 返回统一聚类标签的基础聚类矩阵和稀疏矩阵