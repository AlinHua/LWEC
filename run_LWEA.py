import numpy as np
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, fcluster

def run_lwea(S, ks):
    """
    使用局部加权集成聚类算法（LWEA）

    参数：
    - S: 相似性矩阵即LWCA
    - ks: 聚类数列表

    返回：
    - results_lwea: 聚类结果
    """
    N = S.shape[0]

    d = stod2(S)  # 将相似性矩阵转换为距离向量
    # 平均链接聚类
    Zal = linkage(d, method='average')

    results_lwea = np.zeros((N, len(ks)), dtype=int)
    for i, K in enumerate(ks):
        # print(f'Obtain {K} clusters by LWEA.')
        results_lwea[:, i] = fcluster(Zal, K, criterion='maxclust')  # 生成聚类结果

    return results_lwea

def stod2(S):
    """
    将相似性矩阵转换为距离向量

    参数：
    - S: N x N 的相似性矩阵

    返回：
    - d: 距离向量
    """
    N = S.shape[0]
    s = np.zeros(int(N * (N - 1) / 2))
    next_idx = 0
    for a in range(N - 1):
        s[next_idx:next_idx + (N - a - 1)] = S[a, a + 1:].toarray()  # 将稀疏矩阵转换为密集矩阵
        next_idx += N - a - 1
    d = 1 - s  # 计算距离 d = 1 - 相似性
    return d

# 测试代码
if __name__ == "__main__":
    # 示例相似性矩阵
    S = np.array([
        [1.0, 0.2, 0.2],
        [0.7, 1.0, 0.4],
        [0.2, 0.1, 1.0]
    ])

    # 如果需要使用稀疏矩阵，请取消注释以下行
    S = csr_matrix(S)

    # 聚类数
    cls_nums = [2, 3]

    # 运行 LWEA
    results_lwea = run_lwea(S, cls_nums)
    print("Results of LWEA:")
    print(results_lwea)
