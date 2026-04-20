"""Single molecule processing - standard pipeline"""
import sys
import os
import time
import pickle

sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

from embed.pocket import get_protein_ligand_neighbors, read_smiles, vertices_gen, featurizer_new
import pandas as pd
import numpy as np

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
log_dir = r'E:\pharm3d\one_molecule_run\logs'
os.makedirs(log_dir, exist_ok=True)

def log(msg):
    print(msg)
    with open(os.path.join(log_dir, 'run.log'), 'a') as f:
        f.write(msg + '\n')

log("=== Single Molecule Pipeline Start ===\n")
total_t0 = time.time()

# Step 1: Pocket extraction
log("Step 1: Pocket Extraction")
t0 = time.time()
try:
    pocket_path = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
    log("[OK] Pocket: %s" % pocket_path)
    log("  Time: %.2fs" % (time.time() - t0))
except Exception as e:
    log("[FAIL] Pocket: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Read SMILES
log("\nStep 2: Read SMILES")
t0 = time.time()
min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=1, ncpu=1)
log("[OK] Molecules: %d" % len(df))
log("  Time: %.2fs" % (time.time() - t0))
log("  Bounds: x[%d,%d] y[%d,%d] z[%d,%d]" % (min_x, max_x, min_y, max_y, min_z, max_z))

# Step 3: Vertex generation
log("\nStep 3: Vertex Generation")
t0 = time.time()
vertices = vertices_gen(pocket_path, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=1)
log("[OK] Vertices: %d" % len(vertices))
log("  Time: %.2fs" % (time.time() - t0))

# Step 4: Featurization
log("\nStep 4: Featurization")
t0 = time.time()
ind_att_pd_path, tensor_path, state_path = featurizer_new(
    dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=1
)
log("[OK] Featurization complete")
log("  Time: %.2fs" % (time.time() - t0))
log("  ind_att_pd: %s" % ind_att_pd_path)
log("  tensor.pkl: %s" % tensor_path)
log("  state.csv: %s" % state_path)

# Check outputs
log("\n=== Output Files ===")
for path in [ind_att_pd_path, tensor_path, state_path]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        log("[OK] %s (%d bytes)" % (os.path.basename(path), size))

# Check tensor
log("\n=== Tensor Info ===")
with open(tensor_path, 'rb') as f:
    tensor_data = pickle.load(f)
log("Molecules: %d" % len(tensor_data))
if len(tensor_data) > 0:
    log("Feature dim: %d" % len(tensor_data[0]))

total_time = time.time() - total_t0
log("\n=== Pipeline Complete ===")
log("Total time: %.2fs" % total_time)
