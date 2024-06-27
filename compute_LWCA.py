import numpy as np
import scipy.sparse as sp


def compute_lwca(base_cls_segs, eci, M):
    """
    计算局部加权共同关联矩阵（LWCA）

    参数：
    - base_cls_segs: 表示每个聚类与数据点关系的稀疏矩阵（CSR格式）
    - eci: 熵加权集成指数的np数组
    - M: 基础聚类的数量

    返回：
    - LWCA: 局部加权共同关联矩阵
    """
    # 将 base_cls_segs 转置，行表示数据点，列表示聚类
    base_cls_segs = base_cls_segs.T

    # 获取数据点数量
    N = base_cls_segs.shape[0]

    # 计算 LWCA 矩阵
    lwca = (base_cls_segs.multiply(eci)).dot(base_cls_segs.T) / M

    # 调整对角线元素
    lwca = lwca - sp.diags(lwca.diagonal()) + sp.eye(N)

    return lwca


# 测试代码
if __name__ == "__main__":
    base_cls = np.array([
        [1, 2, 3],
        [2, 3, 1],
        [3, 1, 2]
    ])

    # 示例稀疏矩阵
    data = [1, 1, 1, 1, 1, 1]
    row = [0, 1, 2, 0, 1, 2]
    col = [0, 1, 2, 2, 0, 1]
    base_cls_segs = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    eci = np.array([0.5, 0.8, 0.6])
    M = 3

    lwca = compute_lwca(base_cls_segs, eci, M)
    print("LWCA:")
    print(lwca.toarray())
