import os
import numpy as np
from dataset import Dataset

root = './dataset/ShapeNet/'
num_points = 2048

for split in ['train', 'test']:
    ds = Dataset(root=root,
                 dataset_name='shapenetcorev2',
                 split=split,
                 num_points=num_points,
                 load_name=True,
                 load_file=True,
                 segmentation=False,
                 random_rotate=False,
                 random_jitter=False,
                 random_translate=False)

    out_dir = os.path.join(root, f'{split}_csv')
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(len(ds)):
        pts, label, name, filename = ds[idx]   # filename 可能是 '02992529/xxx.obj'
        pts = pts.numpy()

        # 把 filename 的路径和后缀都去掉，得到相对“类/实例”路径
        rel_path = os.path.splitext(filename)[0]  # e.g. '02992529/71240556...'
        # 生成最终要写入的完整路径
        out_path = os.path.join(out_dir, f'{rel_path}.csv')

        # —— 关键：保证 out_path 的父目录都存在 —— 
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 保存 CSV
        np.savetxt(out_path,
                   pts,
                   delimiter=',',
                   header='x,y,z',
                   comments='')
