# -*- coding: utf-8 -*-
"""
Re-flatten tensor using fixed training index for MLP compatibility
"""
import sys
import os
import pickle
import pandas as pd
import numpy as np

def reflatten_tensor(dirname, fixed_index_path):
    """
    Load generated tensor and re-flatten using fixed training index
    """
    print("=== Tensor Re-flattening ===\n")
    
    # Load fixed training index
    print("Loading fixed index: %s" % fixed_index_path)
    fixed_index = pd.read_csv(fixed_index_path)
    print("  Fixed index shape: %s" % str(fixed_index.shape))
    print("  Unique grid positions: %d" % fixed_index['gridNo'].nunique())
    print("  Feature types: %s" % sorted(fixed_index['featNo'].unique()))
    
    # Load generated tensor
    tensor_path = os.path.join(dirname, "tensor.pkl")
    print("\nLoading generated tensor: %s" % tensor_path)
    with open(tensor_path, 'rb') as f:
        tensors = pickle.load(f)
    print("  Molecules: %d" % len(tensors))
    print("  Feature dim: %d" % len(tensors[0]))
    
    # Use training box bounds (fixed)
    min_x, max_x, min_y, max_y, min_z, max_z = -14, 19, -19, 17, 0, 47
    grid_shape = (max_x - min_x, max_y - min_y, max_z - min_z)
    
    print("  Grid shape: %s" % str(grid_shape))
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    # Create new tensor aligned with fixed index
    num_features = 6  # Donor, Acceptor, PosIonizable, Aromatic, Hydrophobe, LumpedHydrophobe
    
    # Build feature map from fixed index
    # fixed_index has: gridNo, featNo
    # We need to create a mapping: (gridNo, featNo) -> position in flattened vector
    
    # Get unique positions from fixed index
    unique_positions = fixed_index['gridNo'].unique()
    print("\nFixed index covers %d grid positions" % len(unique_positions))
    
    # Create new flattened tensors
    new_tensors = []
    for mol_idx, tensor in enumerate(tensors):
        # Reshape to 3D grid x features
        grid_features = tensor.reshape(total_grids, num_features)
        
        # Extract only the positions and features in fixed_index
        flattened = []
        for _, row in fixed_index.iterrows():
            grid_no = row['gridNo']
            feat_no = row['featNo']
            if grid_no < total_grids and feat_no < num_features:
                flattened.append(grid_features[grid_no, feat_no])
            else:
                flattened.append(0.0)
        
        new_tensors.append(np.array(flattened))
        
        if mol_idx == 0:
            print("  First molecule: extracted %d features" % len(flattened))
    
    # Save re-flattened tensor
    output_path = os.path.join(dirname, "tensor_fixed.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(new_tensors, f)
    
    print("\n[OK] Saved re-flattened tensor: %s" % output_path)
    print("  Size: %d bytes" % os.path.getsize(output_path))
    print("  Shape per molecule: %d" % len(new_tensors[0]))
    
    # Also save as tensor.pkl (overwrite)
    with open(tensor_path, 'wb') as f:
        pickle.dump(new_tensors, f)
    print("  Updated: %s" % tensor_path)
    
    return new_tensors

def main():
    dirname = r'E:\pharm3d\one_molecule_run\job_0001'
    fixed_index_path = r'E:\pharm3d\00011734250639\ind_att_pd.csv'
    
    new_tensors = reflatten_tensor(dirname, fixed_index_path)
    
    print("\n=== Re-flattening Complete ===")
    print("Tensor is now compatible with trained MLP model!")

if __name__ == "__main__":
    main()
