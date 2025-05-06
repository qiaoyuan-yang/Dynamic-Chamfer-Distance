import numpy as np
import pandas as pd

# 读取 .npy 文件
array1 = np.load("/Users/quan/Desktop/tcs-code/Dynamic_Chamfer_Distance/23NIPS/federalist/federalist_84.npy")
array2 = np.load("/Users/quan/Desktop/tcs-code/Dynamic_Chamfer_Distance/23NIPS/federalist/federalist_85.npy")
print(f"npy1 shape: {array1.shape}")
print(f"npy2 shape: {array2.shape}")

# 转换为 pandas DataFrame
df1 = pd.DataFrame(array1)
df2 = pd.DataFrame(array2)

print(f"csv1 rows: {len(df1)}")
print(f"csv2 rows: {len(df2)}")

# 保存为 CSV 文件
df1.to_csv("/Users/quan/Desktop/tcs-code/Dynamic_Chamfer_Distance/dataset/Text_Embeddings/federalist/A.csv", index=False, header=False)
df2.to_csv("/Users/quan/Desktop/tcs-code/Dynamic_Chamfer_Distance/dataset/Text_Embeddings/federalist/B.csv", index=False, header=False)
