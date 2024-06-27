import numpy as np
from scipy.sparse import csr_matrix

def compute_nmi(results, gt):
    """
    计算聚类结果的NMI分数。

    参数:
        results (numpy.ndarray): 聚类结果，每列是一个聚类结果。
        gt (numpy.ndarray): 真实标签。

    返回:
        numpy.ndarray: 每个聚类结果的NMI分数。
    """
    all_scores = np.zeros(results.shape[1])

    for i in range(results.shape[1]):
        # if np.min(results[:, i]) > 0:  # 聚类标签包括0, 因此在计算中此条件应删除
        all_scores[i] = nmi_max(results[:, i], gt)
    return all_scores

def nmi_max(x, y):
    """
    计算两个标签集之间的最大归一化互信息。

    参数:
        x (numpy.ndarray): 第一个标签集。
        y (numpy.ndarray): 第二个标签集。

    返回:
        float: 最大归一化互信息。
    """
    assert len(x) == len(y), "两个标签集的长度必须相等。"

    n = len(x)
    x = x.reshape(n)
    y = y.reshape(n)

    l = min(np.min(x), np.min(y))
    x = x - l + 1
    y = y - l + 1
    k = max(np.max(x), np.max(y))

    idx = np.arange(n)
    Mx = csr_matrix((np.ones(n), (idx, x - 1)), shape=(n, k))
    My = csr_matrix((np.ones(n), (idx, y - 1)), shape=(n, k))
    Pxy = (Mx.T @ My / n).data  # 计算x和y的联合分布
    Hxy = -np.dot(Pxy, np.log(Pxy + np.finfo(float).eps))

    Px = np.mean(Mx, axis=0).A1
    Py = np.mean(My, axis=0).A1

    # 计算Px和Py的熵
    Hx = -np.dot(Px, np.log(Px + np.finfo(float).eps))
    Hy = -np.dot(Py, np.log(Py + np.finfo(float).eps))

    # 互信息
    MI = Hx + Hy - Hxy

    # 最大归一化互信息
    return MI / max(Hx, Hy)

# 示例调用
if __name__ == "__main__":
    # 示例聚类结果矩阵
    results = np.array([
        [1, 2, 1],
        [2, 1, 3],
        [1, 3, 2]
    ])

    # 示例真实标签向量
    gt = np.array([1, 2, 1])

    # 计算 NMI 分数
    all_scores = compute_nmi(results, gt)

    # 打印结果
    print('All NMI Scores:')
    print(all_scores)
