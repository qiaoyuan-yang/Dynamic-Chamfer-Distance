import os
import scipy.io
import pandas as pd

# 当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 构造目标文件绝对路径
mat_path = os.path.join(base_dir, "..", "dataset", "Text_Embeddings", "bbcsport", "bbcsport-emd_tr_te_split.mat")

# 加载 .mat 文件
print(f"Loading: {mat_path}")
mat = scipy.io.loadmat(mat_path)

# 提取文档向量
X = mat["X"]
output_dir = os.path.join(base_dir, "..", "dataset", "Text_Embeddings", "bbcsport_csv")
os.makedirs(output_dir, exist_ok=True)

# 批量保存每个文档
for i in range(X.shape[1]):
    doc = X[0, i].T
    df = pd.DataFrame(doc)
    df.to_csv(os.path.join(output_dir, f"doc_{i}.csv"), index=False, header=False)

print(f"✅ Saved {X.shape[1]} documents to: {output_dir}")
