import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sparse_matrix_generator import generate_sparse_matrix
from compute_ECI import compute_eci

def run_lwgp(bcs, base_cls_segs, eci, cls_num):
    """
    使用局部加权二部图算法进行聚类

    参数：
    - bcs: 基础聚类矩阵
    - base_cls_segs: 表示每个聚类与数据点关系的稀疏矩阵
    - eci: 熵加权集成指数的数组
    - cls_num: 聚类数的列表

    返回：
    - labels: 聚类结果
    """
    # 构建邻接矩阵(N, n_cls)用于表示每个数据点与聚类之间的不确定性关系
    lwA = base_cls_segs.toarray() * eci[:, np.newaxis]

    labels = np.zeros((bcs.shape[0], len(cls_num)), dtype=int)
    for i, num_clusters in enumerate(cls_num):
        # print(f'Obtain {num_clusters} clusters by LWGP.')
        labels[:, i] = bipartite_graph_partitioning(lwA.T, num_clusters)  # 标准化割
    return labels

def bipartite_graph_partitioning(A, Nseg):
    """
    标准化割

    参数：
    - A: 邻接矩阵(N, K)
    - Nseg: 聚类数

    返回：
    - labels: 聚类结果
    """
    Nx, Ny = A.shape  # Nx为数据点数  Ny为聚类数

    if Ny < Nseg:
        raise ValueError('The cluster number is too large!')

    # 计算 U 集的度矩阵(N, N)
    du = np.sum(A, axis=1).flatten()  # 按行求和并转换为一维数组
    du[du == 0] = 1e-10  # 避免后续计算中出现除以零
    Du_inv = diags(1.0 / du).tocsc()  # 转换为压缩稀疏列格式（CSC格式）

    # 计算 V 集的度矩阵(n_cls, n_cls)
    dv = np.sum(A, axis=0).flatten()  # 按列求和并转换为一维数组
    Dv_inv_sqrt = diags(1.0 / np.sqrt(dv)).tocsc()  # 取平方根倒数

    # 构建归一化的相似性矩阵(n_cls, n_cls)
    nWy = Dv_inv_sqrt @ A.T @ Du_inv @ A @ Dv_inv_sqrt

    # 强制对称（如果矩阵可能不完全对称）
    nWy = (nWy + nWy.T) / 2

    # 计算特征值和特征向量
    evals, evecs = eigh(nWy)

    # 选择前Nseg个最小特征值对应的特征向量
    idx = np.argsort(-evals)
    Ncut_evec = Dv_inv_sqrt @ evecs[:, idx[:int(Nseg)]]

    # 将特征向量映射到原始空间，并对每一行进行归一化
    evec = Du_inv @ A @ Ncut_evec

    # 归一化每一行
    evec = evec / (np.sqrt(np.sum(evec ** 2, axis=1, keepdims=True)) + 1e-10)

    # 进行k-means聚类
    kmeans = KMeans(n_clusters=Nseg, max_iter=100, n_init=3, random_state=42)
    labels = kmeans.fit_predict(evec)

    return labels

if __name__ == "__main__":
    base_cls = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
    ])

    bcs, base_cls_segs = generate_sparse_matrix(base_cls)

    # 示例 ECI 向量
    eci = compute_eci(bcs, base_cls_segs, 0.4)
    print("ECI:")
    print(eci)

    # 聚类数
    cls_nums = [2, 3]

    # 运行 LWGP
    results_lwgp = run_lwgp(bcs, base_cls_segs, eci, cls_nums)
    print(results_lwgp)