import numpy as np
import os

dim = 2      # dim
n_A = 100000  # without outlier
n_B = 10000 
outlier_scale = 0.5  # 离群点比例因子
random_seed = 42

# 输出文件路径（根据需要修改目录）
output_dir = './dataset/Gaussian'
file_A = os.path.join(output_dir, 'A.csv')
file_B = os.path.join(output_dir, 'B.csv')

# 创建目录
os.makedirs(output_dir, exist_ok=True)

# 固定随机种子
np.random.seed(random_seed)

# 1. 生成 A, B 高斯点云
A = np.random.randn(n_A, dim)
B = np.random.randn(n_B, dim)

# 2. 向 A 添加离群点：全 1 向量乘以 outlier_scale * dim
outlier = np.ones(dim) * (outlier_scale * n_A)
A = np.vstack([A, outlier])  # A 大小变为 n_A + 1

# 3. 保存到 CSV，每行一个点，逗号分隔
np.savetxt(file_A, A, delimiter=',')
np.savetxt(file_B, B, delimiter=',')

print(f"Generated files:\n  A -> {file_A} (total {A.shape[0]} rows)\n  B -> {file_B} (total {B.shape[0]} rows)")

