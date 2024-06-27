import scipy.io as sio
import numpy as np


def load_and_view_data(file_path):
    set_labels = set()
    try:
        # 加载 .mat 文件
        data = sio.loadmat(file_path)

        # 打印文件中的所有变量名及其数据类型
        print("文件中的变量:")
        for key, value in data.items():
            if not key.startswith('__'):  # 过滤掉__开头特殊变量如__header__,__version__等
                print(f"{key}: {type(value)} with shape {value.shape if isinstance(value, np.ndarray) else 'N/A'}")

        # 打印变量内容的示例
        for key, value in data.items():
            if not key.startswith('__'):  # 过滤掉__开头特殊变量如__header__,__version__等
                print(f"\n{key}:")
                print(value)
            # 打印出所有标签取值
            if key == 'gt':
                for label in value:
                    set_labels.add(label[0])
        print(f"\nRange of labels: {set_labels}")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在。")
    except Exception as e:
        print(f"发生错误: {str(e)}")


# 主程序入口
if __name__ == "__main__":
    data_name = 'VS'
    mat_file_path = f'./Dateset/bc_pool_{data_name}.mat'

    # 调用函数
    load_and_view_data(mat_file_path)
