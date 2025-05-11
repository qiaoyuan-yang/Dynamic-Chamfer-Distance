import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 从文件读取 CSV 数据
# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 cost_log.csv 的绝对路径
# csv_path = os.path.join(script_dir, '..', 'output', 'Gaussion.csv')
csv_path = os.path.join(script_dir, '..', 'output', 'cost_log.csv')

data = pd.read_csv(csv_path)


# 绘图
plt.figure(figsize=(10, 5))
plt.plot(data['step'], data['our_err'].abs(), color='blue', marker='o', linestyle='-', linewidth=1.5, markersize=3, label='Ours')
plt.plot(data['step'], data['uni_err'].abs(), color='orange', marker='s', linestyle='--', linewidth=1.5, markersize=3, label='Uniform')

# plt.title('Gaussion point')
plt.title('text embedding')
# plt.title('MNIST')
plt.xlabel('Time Step')
plt.ylabel('Error (x - chamfer) / chamfer')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.legend(loc='upper right')  

# 保存图像到当前目录
# save_path = os.path.join(os.path.dirname(__file__), 'Gaussion.png')
save_path = os.path.join(os.path.dirname(__file__), 'text_embedding.png')
# save_path = os.path.join(os.path.dirname(__file__), 'MNIST.png')
plt.savefig(save_path, dpi=300)  # 高分辨率输出
plt.close()

print(f"Figure saved: {save_path}")
