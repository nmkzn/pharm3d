# -*- coding: utf-8 -*-
"""
Minimal probe script - verify single molecule processing pipeline
"""
import sys
import os
import time

# Set path
sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

print("=== Stage 1: Module Import Test ===")
try:
    from embed.pocket import get_protein_ligand_neighbors, read_smiles, vertices_gen, featurizer_new
    from embed.utils import getSimplifiedMatrix
    import pandas as pd
    import numpy as np
    import pickle
    print("[OK] All modules imported successfully")
except Exception as e:
    print("[FAIL] Import failed: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
ncpu = 4
num_confs = 5

print("\n=== Stage 2: Check Input Files ===")
required_files = [
    'template_known_mols.smi',
    'crystal_ligand.mol2',
    'template_complex.pdb'
]
for f in required_files:
    path = os.path.join(dirname, f)
    if os.path.exists(path):
        print("[OK] %s" % f)
    else:
        print("[FAIL] %s missing" % f)
        sys.exit(1)

print("\n=== Stage 3: Check Ligand Residue in PDB ===")
try:
    with open(os.path.join(dirname, 'template_complex.pdb'), 'r') as f:
        pdb_content = f.read()
    if 'UNK' in pdb_content:
        print("[OK] Found UNK residue (ligand)")
    else:
        print("[WARN] UNK not found, may need to adjust ligand_residue_id")
        residues = set()
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                resn = line[17:20].strip()
                residues.add(resn)
        print("  Found residues: %s" % sorted(residues)[:10])
except Exception as e:
    print("[FAIL] PDB check failed: %s" % e)

print("\n=== Stage 4: Pocket Extraction Test ===")
try:
    t0 = time.time()
    pocket_path_npy = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
    t1 = time.time()
    print("[OK] Pocket extraction: %.2fs" % (t1-t0))
    print("  Output: %s" % pocket_path_npy)
    
    if os.path.exists(pocket_path_npy):
        data = np.load(pocket_path_npy, allow_pickle=True)
        print("  Pocket data shape: %s" % str(data.shape))
except Exception as e:
    print("[FAIL] Pocket extraction failed: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Stage 5: SMILES Reading and Conformer Generation ===")
try:
    t0 = time.time()
    min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=num_confs, ncpu=ncpu)
    t1 = time.time()
    print("[OK] Conformer generation: %.2fs" % (t1-t0))
    print("  Generated molecules: %d" % len(df))
    print("  Grid bounds: x[%d,%d] y[%d,%d] z[%d,%d]" % (min_x, max_x, min_y, max_y, min_z, max_z))
except Exception as e:
    print("[FAIL] Conformer generation failed: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Stage 6: Vertex Generation ===")
try:
    t0 = time.time()
    vertices = vertices_gen(pocket_path_npy, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=ncpu)
    t1 = time.time()
    print("[OK] Vertex generation: %.2fs" % (t1-t0))
    print("  Vertex count: %d" % len(vertices))
except Exception as e:
    print("[FAIL] Vertex generation failed: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Stage 7: Featurization and Tensor Generation ===")
try:
    t0 = time.time()
    ind_att_pd_path, tensor_path, state_path = featurizer_new(
        dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=ncpu
    )
    t1 = time.time()
    print("[OK] Featurization: %.2fs" % (t1-t0))
    print("  ind_att_pd: %s" % ind_att_pd_path)
    print("  tensor.pkl: %s" % tensor_path)
    print("  state.csv: %s" % state_path)
except Exception as e:
    print("[FAIL] Featurization failed: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Stage 8: Verify Output ===")
for path in [ind_att_pd_path, tensor_path, state_path]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print("[OK] %s (%d bytes)" % (os.path.basename(path), size))
    else:
        print("[FAIL] %s not found" % path)

print("\n=== Stage 9: Check Tensor Dimensions ===")
try:
    with open(tensor_path, 'rb') as f:
        tensor_data = pickle.load(f)
    print("[OK] tensor.pkl contains %d molecules" % len(tensor_data))
    if len(tensor_data) > 0:
        print("  Feature dimension per molecule: %d" % len(tensor_data[0]))
except Exception as e:
    print("[FAIL] Tensor read failed: %s" % e)

print("\n=== Minimal Probe Complete ===")
