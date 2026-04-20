# -*- coding: utf-8 -*-
"""Step 1: Test imports and pocket extraction only"""
import sys
import os
import time

# Set PyMOL to headless mode before importing
os.environ['PYMOL_HEADLESS'] = '1'

sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

print("=== Stage 1: Module Import ===")
try:
    # Import pymol with headless setup
    import pymol
    pymol.finish_launching(['pymol', '-qc'])  # quiet, no GUI
    from embed.pocket import get_protein_ligand_neighbors
    print("[OK] Import successful")
except Exception as e:
    print("[FAIL] Import: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

dirname = r'E:\pharm3d\one_molecule_run\job_0001'

print("\n=== Stage 2: Pocket Extraction ===")
try:
    t0 = time.time()
    pocket_path_npy = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
    t1 = time.time()
    print("[OK] Pocket extraction: %.2fs" % (t1-t0))
    print("  Output: %s" % pocket_path_npy)
    
    if os.path.exists(pocket_path_npy):
        import numpy as np
        data = np.load(pocket_path_npy, allow_pickle=True)
        print("  Pocket data shape: %s" % str(data.shape))
        print("\n=== Step 1 Complete ===")
except Exception as e:
    print("[FAIL] Pocket extraction: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
