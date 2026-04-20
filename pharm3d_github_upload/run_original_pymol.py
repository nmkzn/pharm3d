# -*- coding: utf-8 -*-
"""Run original PyMOL version of pocket extraction"""
import os
import sys
import time

# 设置headless模式
os.environ['PYMOL_HEADLESS'] = '1'
os.environ['PYMOL_QUIET'] = '1'
os.environ['DISPLAY'] = ''

sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

from embed.pocket import get_protein_ligand_neighbors, read_smiles, vertices_gen, featurizer_new

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
log_file = os.path.join(dirname, 'original_pymol.log')

def log(msg):
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

log("=== Original PyMOL Pipeline ===\n")
log("PyMOL Version Test: Success")

# Step 1: Original PyMOL pocket extraction
log("Step 1: Pocket Extraction (Original PyMOL)")
t0 = time.time()
try:
    pocket_path = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
    t1 = time.time()
    log("[OK] Pocket: %s" % pocket_path)
    log("  Time: %.2fs" % (t1 - t0))
except Exception as e:
    log("[FAIL] Pocket: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Read SMILES (same as before)
log("\nStep 2: Read SMILES")
t0 = time.time()
min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=1, ncpu=1)
t1 = time.time()
log("[OK] Molecules: %d" % len(df))
log("  Time: %.2fs" % (t1 - t0))

# Step 3: Vertex Generation (same as before)
log("\nStep 3: Vertex Generation")
t0 = time.time()
vertices = vertices_gen(pocket_path, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=1)
t1 = time.time()
log("[OK] Vertices: %d" % len(vertices))
log("  Time: %.2fs" % (t1 - t0))

# Step 4: Featurization (same as before)
log("\nStep 4: Featurization")
t0 = time.time()
ind_att_pd_path, tensor_path, state_path = featurizer_new(dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=1)
t1 = time.time()
log("[OK] Featurization complete")
log("  Time: %.2fs" % (t1 - t0))

log("\n=== Original PyMOL Pipeline Complete ===")
log("Output: %s" % tensor_path)
