import numpy as np
from sklearn.metrics import adjusted_rand_score


def compute_ari(results, gt):
    """
    计算调整后的兰德指数（ARI）

    参数:
        results (numpy.ndarray): 聚类结果，每列是一个聚类结果。
        gt (numpy.ndarray): 真实标签 (列向量)。

    返回:
        numpy.ndarray: 每个聚类结果的ARI分数。
    """
    if results.shape[0] != gt.shape[0]:
        raise ValueError("每个聚类结果的数量必须与真实标签的数量相同。")

    # 初始化一个数组来存储每个聚类结果的ARI分数
    ari_scores = np.zeros(results.shape[1])

    # 计算每个聚类结果的ARI分数
    for i in range(results.shape[1]):
        ari_scores[i] = adjusted_rand_score(gt, results[:, i])

    return ari_scores


if __name__ == "__main__":
    # 示例数据
    gt = np.array([0, 0, 1, 1, 2, 2])
    results = np.array([
        [0, 0, 2, 1, 2, 2],
        [0, 0, 1, 1, 1, 1]
    ]).T  # 每列是一个聚类结果

    ari_scores = compute_ari(results, gt)
    print(f"ARI scores: {ari_scores}")
