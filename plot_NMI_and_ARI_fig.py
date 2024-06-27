import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from compute_ECI import compute_eci
from compute_LWCA import compute_lwca
from run_LWGP import run_lwgp
from run_LWEA import run_lwea
from compute_NMI import compute_nmi
from compute_ARI import compute_ari
from sparse_matrix_generator import generate_sparse_matrix

# 定义一个函数，用于计算和绘制给定数据集的NMI分数（运行20次的平均结果）
def plot_nmi_and_ari(dataset_name, ax_nmi, ax_ari, plot_idx, runs=20):
    # 加载数据集
    data = sio.loadmat(f'./Dateset/bc_pool_{dataset_name}.mat')
    members = data['members']
    gt = data['gt'].flatten()

    N, pool_size = members.shape
    theta = 0.4
    cls_nums = np.arange(2, 33)  # 聚类数目范围
    ensemble_sizes = np.arange(10, 51, 10)  # 集成大小范围

    # 初始化数组以存储平均NMI分数
    avg_nmi_scores_lwea = np.zeros(len(ensemble_sizes))
    avg_nmi_scores_lwgp = np.zeros(len(ensemble_sizes))

    # 初始化数组以存储平均ARI分数
    avg_ari_scores_lwea = np.zeros(len(ensemble_sizes))
    avg_ari_scores_lwgp = np.zeros(len(ensemble_sizes))

    for run in range(runs):
        print(f"Run {run}:\n")
        # 初始化数组以存储平均NMI分数
        run_nmi_scores_lwea = np.zeros(len(ensemble_sizes))
        run_nmi_scores_lwgp = np.zeros(len(ensemble_sizes))

        # 初始化数组以存储平均ARI分数
        run_ari_scores_lwea = np.zeros(len(ensemble_sizes))
        run_ari_scores_lwgp = np.zeros(len(ensemble_sizes))

        for i, ensemble_size in enumerate(ensemble_sizes):
            print(f"Ensemble_size {ensemble_size} \n")
            # 随机选择ensemble_size个基础聚类
            bc_idx = np.random.choice(pool_size, ensemble_size, replace=False)
            base_cls = members[:, bc_idx]

            # 生成稀疏矩阵
            bcs, base_cls_segs = generate_sparse_matrix(base_cls)

            # 计算ECI
            eci = compute_eci(bcs, base_cls_segs, theta)

            # 计算LWCA
            lwca = compute_lwca(base_cls_segs, eci, ensemble_size)

            # 运行LWGP和LWEA
            results_lwgp = run_lwgp(bcs, base_cls_segs, eci, cls_nums)
            results_lwea = run_lwea(lwca, cls_nums)

            # 计算NMI分数
            nmi_scores_lwgp = compute_nmi(results_lwgp, gt)
            nmi_scores_lwea = compute_nmi(results_lwea, gt)

            # 计算ARI分数
            ari_scores_lwgp = compute_ari(results_lwgp, gt)
            ari_scores_lwea = compute_ari(results_lwea, gt)

            # 最佳簇数下的NMI分数
            run_nmi_scores_lwea[i] = np.max(nmi_scores_lwgp)
            run_nmi_scores_lwgp[i] = np.max(nmi_scores_lwea)

            # 最佳簇数下的ARI分数
            run_ari_scores_lwea[i] = np.max(ari_scores_lwgp)
            run_ari_scores_lwgp[i] = np.max(ari_scores_lwea)

        avg_nmi_scores_lwea += run_nmi_scores_lwea
        avg_nmi_scores_lwgp += run_nmi_scores_lwgp

        avg_ari_scores_lwea += run_ari_scores_lwea
        avg_ari_scores_lwgp += run_ari_scores_lwgp

    avg_nmi_scores_lwea /= runs
    avg_nmi_scores_lwgp /= runs

    avg_ari_scores_lwea /= runs
    avg_ari_scores_lwgp /= runs

    # 绘制NMI分数
    ax_nmi.plot(ensemble_sizes, avg_nmi_scores_lwea, label='LWEA', linestyle='-', color='blue', marker='^')
    ax_nmi.plot(ensemble_sizes, avg_nmi_scores_lwgp, label='LWGP', linestyle='--', color='red', marker='o')

    ax_nmi.set_title(f'({chr(97 + plot_idx)}) {dataset_name}')
    ax_nmi.set_xlabel('Ensemble Size')
    ax_nmi.set_ylabel('NMI')
    ax_nmi.legend()
    ax_nmi.set_ylim(0, 1)  # 限制纵坐标范围在0到1之间以便美观

    # 绘制ARI分数
    ax_ari.plot(ensemble_sizes, avg_ari_scores_lwea, label='LWEA', linestyle='-', color='blue', marker='^')
    ax_ari.plot(ensemble_sizes, avg_ari_scores_lwgp, label='LWGP', linestyle='--', color='red', marker='o')

    ax_ari.set_title(f'({chr(97 + plot_idx)}) {dataset_name}')
    ax_ari.set_xlabel('Ensemble Size')
    ax_ari.set_ylabel('ARI')
    ax_ari.legend()
    ax_ari.set_ylim(0, 1)  # 限制纵坐标范围在0到1之间以便美观


# 数据集名称
dataset_names = ["Caltech20", "FCT", "IS", "ISOLET", "LR", "LS", "MF", "MNIST", "ODR", "PD", "Semeion", "SPF",
               "Texture", "VS", "USPS"]

# dataset_names = ["Caltech20"]
# 创建一个3x5的子图布局
fig_nmi, axes_nmi = plt.subplots(3, 5, figsize=(16, 12))
fig_ari, axes_ari = plt.subplots(3, 5, figsize=(16, 12))

for plot_idx, dataset_name in enumerate(dataset_names):
    row, col = divmod(plot_idx, 5)
    print(f"Dateset: {dataset_name}\n")
    plot_nmi_and_ari(dataset_name, axes_nmi[row, col], axes_ari[row, col], plot_idx)

# 调整布局
fig_nmi.tight_layout()
fig_ari.tight_layout()
fig_nmi.savefig('./Result/LWEA_and_LWGP_NMI_Result.pdf')
fig_ari.savefig('./Result/LWEA_and_LWGP_ARI_Result.pdf')
