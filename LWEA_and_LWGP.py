import numpy as np
import scipy.io as sio
from sklearn.metrics import normalized_mutual_info_score
import random
import time
from sparse_matrix_generator import generate_sparse_matrix
from compute_ECI import compute_eci
from compute_LWCA import compute_lwca
from run_LWGP import run_lwgp
from run_LWEA import run_lwea
from compute_NMI import compute_nmi
from compute_ARI import compute_ari


"""
Locally Weighted Ensemble Clustering for the LWEA and LWGP algorithms.
"""

def run_LWEA_and_LWGP_avgNMIandARI(data_name, theta, M, cnt_times):
    # 1. 加载数据集
    data_name = data_name
    data = sio.loadmat(f'./Dateset/bc_pool_{data_name}.mat')
    members = data['members']  # 提取基础聚类池(2000, 100)即2000个数据点，100个基础聚类，每一行代表着该数据点在100个基础聚类中的标签
    gt = data['gt'].flatten()  # 提取真实标签(2000,1)即2000个数据点的真实聚类标签，并展平为1D向量

    N, pool_size = members.shape  # N为数据个数，pool_size为基础聚类池的个数

    # 2. 设置重复实验，每次从基础聚类池中随机选择 M 个基础聚类来进行集成聚类
    bc_idx = np.zeros((cnt_times, M), dtype=int)  # 存储每次运行中随机选择的基础聚类的索引,每一行代表一次实验
    for i in range(cnt_times):
        bc_idx[i, :] = np.random.choice(pool_size, M, replace=False)

    # 3. 存储实验结果
    cls_nums = np.arange(2, 31)  # 设置聚类的数量范围
    nmi_scores_bestk_lwea = np.zeros(cnt_times)  # 存储每次运行中 LWEA 算法在最佳(NMI最大)聚类数目下的NMI
    nmi_scores_truek_lwea = np.zeros(cnt_times)  # 存储每次运行中 LWEA 算法在真实(gt)聚类数目下的NMI
    nmi_scores_bestk_lwgp = np.zeros(cnt_times)  # 存储每次运行中 LWGP 算法在最佳聚类数目下的NMI
    nmi_scores_truek_lwgp = np.zeros(cnt_times)  # 存储每次运行中 LWGP 算法在真实聚类数目下的NMI

    ari_scores_bestk_lwea = np.zeros(cnt_times)  # 存储每次运行中 LWEA 算法在最佳(NMI最大)聚类数目下的ARI
    ari_scores_truek_lwea = np.zeros(cnt_times)  # 存储每次运行中 LWEA 算法在真实(gt)聚类数目下的ARI
    ari_scores_bestk_lwgp = np.zeros(cnt_times)  # 存储每次运行中 LWGP 算法在最佳聚类数目下的ARI
    ari_scores_truek_lwgp = np.zeros(cnt_times)  # 存储每次运行中 LWGP 算法在真实聚类数目下的ARI

    # 4. 主循环计算结果
    for run_idx in range(cnt_times):
        print('╔══════════════════════════════════════════════════════════════╗')
        print(f'║                          Run {run_idx + 1}:                              ║')
        print('╚══════════════════════════════════════════════════════════════╝')

        # 提取本次实验模拟的基础聚类
        base_cls = members[:, bc_idx[run_idx, :]]  # (维度: (2000, 10))

        # 获取统一聚类标签的后的基础聚类矩阵以及生成稀疏矩阵
        bcs, base_cls_segs = generate_sparse_matrix(base_cls)

        # 计算 ECI
        print('Compute ECI ...')
        start_time = time.time()
        eci = compute_eci(bcs, base_cls_segs, theta)
        print(f'Time taken: {time.time() - start_time}s')

        # 计算 LWCA
        lwca = compute_lwca(base_cls_segs, eci, M)

        # LWGP
        print('Run the LWGP algorithm ...')
        results_lwgp = run_lwgp(bcs, base_cls_segs, eci, cls_nums)
        print('──────────────────────────────────────────────────────────────')

        # LWEA
        print('Run the LWEA algorithm ...')
        results_lwea = run_lwea(lwca, cls_nums)
        print('──────────────────────────────────────────────────────────────')

        # 展示运行结果
        print('╔══════════════════════════════════════════════════════════════╗')
        scores_lwgp = compute_nmi(results_lwgp, gt)
        scores_lwea = compute_nmi(results_lwea, gt)
        ari_lwgp = compute_ari(results_lwgp, gt)
        ari_lwea = compute_ari(results_lwea, gt)

        # 分别采用两种标准来指定共识聚类的簇数
        nmi_scores_bestk_lwea[run_idx] = np.max(scores_lwea)
        true_k = len(np.unique(gt))
        nmi_scores_truek_lwea[run_idx] = scores_lwea[cls_nums == true_k]

        nmi_scores_bestk_lwgp[run_idx] = np.max(scores_lwgp)
        nmi_scores_truek_lwgp[run_idx] = scores_lwgp[cls_nums == true_k]

        ari_scores_bestk_lwea[run_idx] = np.max(ari_lwea)
        ari_scores_truek_lwea[run_idx] = ari_lwea[cls_nums == true_k]

        ari_scores_bestk_lwgp[run_idx] = np.max(ari_lwgp)
        ari_scores_truek_lwgp[run_idx] = ari_lwgp[cls_nums == true_k]

        print(f'The Scores at Run {run_idx + 1}')
        print('    ──────────── The NMI scores w.r.t. best-k: ────────────    ')
        print(f'LWGP : {nmi_scores_bestk_lwgp[run_idx]}')
        print(f'LWEA : {nmi_scores_bestk_lwea[run_idx]}')

        print('    ──────────── The NMI scores w.r.t. true-k: ────────────    ')
        print(f'LWGP : {nmi_scores_truek_lwgp[run_idx]}')
        print(f'LWEA : {nmi_scores_truek_lwea[run_idx]}')

        print(f'The Scores at Run {run_idx + 1}')
        print('    ──────────── The ARI scores w.r.t. best-k: ────────────    ')
        print(f'LWGP : {ari_scores_bestk_lwgp[run_idx]}')
        print(f'LWEA : {ari_scores_bestk_lwea[run_idx]}')

        print('    ──────────── The ARI scores w.r.t. true-k: ────────────    ')
        print(f'LWGP : {ari_scores_truek_lwgp[run_idx]}')
        print(f'LWEA : {ari_scores_truek_lwea[run_idx]}')

        print('╚══════════════════════════════════════════════════════════════╝')

    # 输出结果
    print('╔══════════════════════════════════════════════════════════════╗')
    print(f'** Average Performance over {cnt_times} runs on the {data_name} dataset **')
    print(f'Data size: {N}')
    print(f'Ensemble size: {M}')
    print('    ──────────── Average NMI scores w.r.t. best-k: ────────────    ')
    print(f'LWGP   : {np.mean(nmi_scores_bestk_lwgp)}')
    print(f'LWEA   : {np.mean(nmi_scores_bestk_lwea)}')
    print('    ──────────── Average NMI scores w.r.t. true-k: ────────────    ')
    print(f'LWGP   : {np.mean(nmi_scores_truek_lwgp)}')
    print(f'LWEA   : {np.mean(nmi_scores_truek_lwea)}')
    print('    ──────────── Average ARI scores w.r.t. best-k: ────────────    ')
    print(f'LWGP   : {np.mean(ari_scores_bestk_lwgp)}')
    print(f'LWEA   : {np.mean(ari_scores_bestk_lwea)}')
    print('    ──────────── Average ARI scores w.r.t. true-k: ────────────    ')
    print(f'LWGP   : {np.mean(ari_scores_truek_lwgp)}')
    print(f'LWEA   : {np.mean(ari_scores_truek_lwea)}')
    print('╚══════════════════════════════════════════════════════════════╝')

# 探究不同参数θ下的LWEA和LWGP方法NMI评估
def run_LWEA_and_LWGP_theta(data_name, theta, M, cnt_times):
    # 1. 加载数据集
    data_name = data_name
    data = sio.loadmat(f'./Dateset/bc_pool_{data_name}.mat')
    members = data['members']  # 提取基础聚类池(2000, 100)即2000个数据点，100个基础聚类，每一行代表着该数据点在100个基础聚类中的标签
    gt = data['gt'].flatten()  # 提取真实标签(2000,1)即2000个数据点的真实聚类标签，并展平为1D向量

    N, pool_size = members.shape  # N为数据个数，pool_size为基础聚类池的个数

    # 2. 设置重复实验，每次从基础聚类池中随机选择 M 个基础聚类来进行集成聚类
    bc_idx = np.zeros((cnt_times, M), dtype=int)  # 存储每次运行中随机选择的基础聚类的索引,每一行代表一次实验
    for i in range(cnt_times):
        bc_idx[i, :] = np.random.choice(pool_size, M, replace=False)

    # 3. 存储实验结果
    cls_range = np.arange(2, np.sqrt(N))  # 设置聚类的数量范围
    cls_nums = np.random.choice(cls_range, size=20, replace=False).astype(int)  # 随机选择20个聚类数量
    # 4. 主循环计算结果
    for run_idx in range(cnt_times):
        print('╔══════════════════════════════════════════════════════════════╗')
        print(f'║                          Run {run_idx + 1}:                              ║')
        print('╚══════════════════════════════════════════════════════════════╝')

        # 提取本次实验模拟的基础聚类
        base_cls = members[:, bc_idx[run_idx, :]]  # (维度: (2000, 10))

        # 获取统一聚类标签的后的基础聚类矩阵以及生成稀疏矩阵
        bcs, base_cls_segs = generate_sparse_matrix(base_cls)

        # 计算 ECI
        print('Compute ECI ...')
        start_time = time.time()
        eci = compute_eci(bcs, base_cls_segs, theta)
        print(f'Time taken: {time.time() - start_time}s')

        # 计算 LWCA
        lwca = compute_lwca(base_cls_segs, eci, M)

        # LWGP
        print('Run the LWGP algorithm ...')
        results_lwgp = run_lwgp(bcs, base_cls_segs, eci, cls_nums)
        print('──────────────────────────────────────────────────────────────')

        # LWEA
        print('Run the LWEA algorithm ...')
        results_lwea = run_lwea(lwca, cls_nums)
        print('──────────────────────────────────────────────────────────────')

        # 展示运行结果
        print('╔══════════════════════════════════════════════════════════════╗')
        scores_lwgp = compute_nmi(results_lwgp, gt)
        scores_lwea = compute_nmi(results_lwea, gt)

        # 不同参数θ值下的LWEA和LWGP方法性能
        print()
        print(f'    ──────────── The average NMI scores (theta={theta}) base on {data_name}: ────────────    ')
        print(f'LWGP : {scores_lwgp.mean()}')
        print(f'LWEA : {scores_lwea.mean()}')
        print('╚══════════════════════════════════════════════════════════════╝')

if __name__ == "__main__":
    # 数据集
    data_names = ["MF", "Caltech20", "FCT", "IS", "ISOLET", "LR", "LS", "MNIST", "ODR", "PD", "Semeion", "SPF", "Texture", "USPS", "VS"]
    # 设置参数
    thetas = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4]
    M = 10  # 基础聚类数量
    cnt_times = 1  # 实验次数
    np.random.seed(42)  # 设置随机种子

    # # 探究不同参数θ下的LWEA和LWGP方法NMI评估
    # for data_name in data_names:
    #     for theta in thetas:
    #         run_LWEA_and_LWGP_theta(data_name, theta, M, cnt_times)

    # 探究指定最佳簇数以及真实簇数时方法的性能
    for data_name in data_names:
        run_LWEA_and_LWGP_avgNMIandARI(data_name, 0.4, M, 10)

