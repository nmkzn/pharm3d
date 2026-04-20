"""Minimal probe - test imports and basic functions"""
import sys
import os
import time

sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

log_dir = r'E:\pharm3d\one_molecule_run\logs'
os.makedirs(log_dir, exist_ok=True)

def log(msg):
    print(msg)
    with open(os.path.join(log_dir, 'probe.log'), 'a') as f:
        f.write(msg + '\n')

log("=== Stage 1: Module Import ===")
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
    log("[OK] Core modules imported")
except Exception as e:
    log("[FAIL] Core import: %s" % e)
    sys.exit(1)

try:
    from embed.pocket import read_smiles, vertices_gen, featurizer_new
    log("[OK] Pocket modules imported")
except Exception as e:
    log("[WARN] Pocket import: %s" % e)
    log("  Will try direct RDKit approach")

dirname = r'E:\pharm3d\one_molecule_run\job_0001'

# Check input files
log("\n=== Stage 2: Input Files ===")
files = ['template_known_mols.smi', 'crystal_ligand.mol2', 'template_complex.pdb', 'box_info.txt']
for f in files:
    path = os.path.join(dirname, f)
    if os.path.exists(path):
        log("[OK] %s" % f)
    else:
        log("[MISSING] %s" % f)

# Check PDB for ligand
log("\n=== Stage 3: PDB Ligand Check ===")
try:
    with open(os.path.join(dirname, 'template_complex.pdb'), 'r') as f:
        content = f.read()
    if 'UNK' in content:
        log("[OK] Found UNK residue")
    else:
        log("[WARN] UNK not found, checking residues...")
        residues = set()
        for line in content.split('\n')[:100]:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                resn = line[17:20].strip()
                residues.add(resn)
        log("  Residues found: %s" % sorted(residues)[:10])
except Exception as e:
    log("[FAIL] PDB check: %s" % e)

# Test read_smiles
log("\n=== Stage 4: read_smiles Test ===")
try:
    t0 = time.time()
    from embed.pocket import read_smiles
    min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=1, ncpu=1)
    t1 = time.time()
    log("[OK] read_smiles: %.2fs" % (t1-t0))
    log("  Molecules: %d" % len(df))
    log("  Bounds: x[%d,%d] y[%d,%d] z[%d,%d]" % (min_x, max_x, min_y, max_y, min_z, max_z))
except Exception as e:
    log("[FAIL] read_smiles: %s" % e)
    import traceback
    traceback.print_exc()

log("\n=== Probe Complete ===")
