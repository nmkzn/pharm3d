"""Re-flatten tensor with fixed training index"""
import os
import pickle
import pandas as pd
import numpy as np

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
fixed_index_path = r'E:\pharm3d\00011734250639\ind_att_pd.csv'
log_dir = r'E:\pharm3d\one_molecule_run\logs'

def log(msg):
    print(msg)
    with open(os.path.join(log_dir, 'reflatten.log'), 'a') as f:
        f.write(msg + '\n')

log("=== Tensor Re-flattening ===\n")

# Load fixed index
log("Loading fixed index...")
fixed_index = pd.read_csv(fixed_index_path)
log("  Index entries: %d" % len(fixed_index))
log("  Unique grid positions: %d" % fixed_index['gridNo'].nunique())

# Load generated tensor
log("\nLoading tensor.pkl...")
tensor_path = os.path.join(dirname, "tensor.pkl")
with open(tensor_path, 'rb') as f:
    tensors = pickle.load(f)
log("  Molecules: %d" % len(tensors))
log("  Original dim: %d" % len(tensors[0]))

# Get grid shape from box_info
with open(os.path.join(dirname, 'box_info.txt'), 'r') as f:
    bounds = [int(x) for x in f.read().strip().split()]
min_x, max_x, min_y, max_y, min_z, max_z = bounds
grid_shape = (max_x - min_x, max_y - min_y, max_z - min_z)
total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
log("  Grid shape: %s (total: %d)" % (str(grid_shape), total_grids))

# Re-flatten
feats_dic = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
             "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}
num_features = len(feats_dic)

new_tensors = []
for tensor in tensors:
    grid_features = tensor.reshape(total_grids, num_features)
    flattened = []
    for _, row in fixed_index.iterrows():
        grid_no = row['gridNo']
        feat_no = row['featNo']
        if grid_no < total_grids and feat_no < num_features:
            flattened.append(grid_features[grid_no, feat_no])
        else:
            flattened.append(0.0)
    new_tensors.append(np.array(flattened))

log("\nNew tensor dim: %d" % len(new_tensors[0]))

# Save
with open(tensor_path, 'wb') as f:
    pickle.dump(new_tensors, f)
log("\n[OK] Updated: %s" % tensor_path)
log("  Size: %d bytes" % os.path.getsize(tensor_path))

log("\n=== Re-flattening Complete ===")
log("Tensor is now compatible with trained MLP!")
