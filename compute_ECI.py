import numpy as np
import scipy.sparse as sp

def compute_eci(bcs, base_cls_segs, para_theta):
    """
    计算熵加权集成指数（ECI）

    参数：
    - bcs: 基础聚类矩阵，每列表示一个基础聚类，每行表示一个数据点
    - base_cls_segs: 表示每个聚类与数据点关系的稀疏矩阵，二值矩阵，若第i个数据点属于第j个聚类，则S[j-1][i]=1反之为0
    - para_theta: 参数，用于计算ECI

    返回：
    - ECI: 以np数组形式返回每个聚类的熵加权集成指数
    """
    M = bcs.shape[1]  # 基础聚类数量
    ETs = get_all_cls_entropy(bcs, base_cls_segs)  # 计算每个聚类C_i相对于整个基础聚类的熵
    ECI = np.exp(-ETs / para_theta / M)  # 计算ECI
    return ECI

def get_all_cls_entropy(bcs, base_cls_segs):
    """
    计算每个聚类的相对于整个基础聚类的熵

    参数：
    - bcs: 基础聚类矩阵
    - base_cls_segs: 表示每个聚类与数据点关系的稀疏矩阵

    返回：
    - Es: 以np数组的形式返回每个聚类的熵
    """
    base_cls_segs = base_cls_segs.T  # 行为数据点，列为聚类
    n_cls = base_cls_segs.shape[1]  # 获取聚类总数

    Es = np.zeros(n_cls)
    for i in range(n_cls):
        part_bcs = bcs[base_cls_segs[:, i].toarray().flatten() != 0, :]  # 使用布尔数组对基础聚类矩阵 bcs 进行行索引来提取属于第i个聚类的数据点
        Es[i] = get_one_cls_entropy(part_bcs)  # 计算每个聚类相较于所有基础聚类上的熵
    return Es

def get_one_cls_entropy(part_bcs):
    """
    计算一个聚类在所有基础聚类上的熵

    参数：
    - part_bcs: 属于第i个聚类的数据点的部分bcs矩阵

    返回：
    - E: 一个聚类的熵
    """
    E = 0
    # 遍历所有的基础聚类
    for i in range(part_bcs.shape[1]):
        # 获取该基础聚类中的标签
        tmp = np.sort(part_bcs[:, i])
        u_tmp = np.unique(tmp)

        # 如果标签为1，则确定即熵为0
        if len(u_tmp) <= 1:
            continue

        # 否则，计算聚类C_i相较于该基础聚类的熵
        cnts = np.zeros(len(u_tmp))
        for j in range(len(u_tmp)):
            cnts[j] = np.sum(tmp == u_tmp[j])  # 计算聚类C_i与C_j^m(基础聚类m中的聚类j)的交集

        cnts = cnts / np.sum(cnts)
        E -= np.sum(cnts * np.log2(cnts))  # 求和
    return E


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

    theta = 1.0
    eci = compute_eci(base_cls, base_cls_segs, theta)
    print("ECI:")
    print(eci)