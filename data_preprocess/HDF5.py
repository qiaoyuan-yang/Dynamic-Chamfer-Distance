# import h5py

# # file_path = './dataset/MNIST/mnist-784-euclidean.hdf5'
# file_path = './dataset/SIFT/sift-128-euclidean.hdf5'
# with h5py.File(file_path, 'r') as f:
#     def print_name(name):
#         print(name)
#     f.visit(print_name)

import h5py
import pandas as pd
import os

# Path to your HDF5 file
# file_path = './dataset/MNIST/mnist-784-euclidean.hdf5'
file_path = './dataset/SIFT/sift-128-euclidean.hdf5'

# Output directory
# output_dir = './dataset/MNIST'
output_dir = './dataset/SIFT'
os.makedirs(output_dir, exist_ok=True)

# Open the HDF5 file and extract 'test' and 'train' datasets
with h5py.File(file_path, 'r') as f:
    # If 'test' and 'train' are datasets:
    test_data = f['test'][:]
    train_data = f['train'][:]

# Convert to DataFrame and save as CSV
pd.DataFrame(test_data).to_csv(os.path.join(output_dir, 'test.csv'), index=False, header=False)
pd.DataFrame(train_data).to_csv(os.path.join(output_dir, 'train.csv'), index=False, header=False)

print(f"Exported:\n  test -> {os.path.join(output_dir, 'test.csv')}\n  train -> {os.path.join(output_dir, 'train.csv')}")