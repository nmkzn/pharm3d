# -*- coding: utf-8 -*-
"""RDKit-only probe - bypass PyMOL"""
import sys
import os
import time

sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

print("=== Testing RDKit-based pipeline ===")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
    print("[OK] RDKit imported")
except Exception as e:
    print("[FAIL] RDKit import: %s" % e)
    sys.exit(1)

dirname = r'E:\pharm3d\one_molecule_run\job_0001'

# Test 1: Read SMILES
print("\n=== Test 1: Read SMILES ===")
try:
    smiles_path = os.path.join(dirname, "template_known_mols.smi")
    df = pd.read_csv(smiles_path, header=None, names=['smiles', 'id', 'label'])
    print("[OK] Read %d molecules" % len(df))
    print("  First: %s (ID: %s)" % (df.iloc[0]['smiles'], df.iloc[0]['id']))
except Exception as e:
    print("[FAIL] %s" % e)
    import traceback
    traceback.print_exc()

# Test 2: Generate 3D conformer
print("\n=== Test 2: Generate 3D Conformer ===")
try:
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(df.iloc[0]['smiles'])
    mol = Chem.AddHs(mol)
    
    t0 = time.time()
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    t1 = time.time()
    
    print("[OK] Conformer generation: %.2fs" % (t1-t0))
    
    # Get coordinates
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    print("  Atoms: %d" % mol.GetNumAtoms())
    print("  Bounds: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]" % (
        positions[:,0].min(), positions[:,0].max(),
        positions[:,1].min(), positions[:,1].max(),
        positions[:,2].min(), positions[:,2].max()
    ))
except Exception as e:
    print("[FAIL] %s" % e)
    import traceback
    traceback.print_exc()

# Test 3: Load reference
print("\n=== Test 3: Load Reference ===")
try:
    ref_path = os.path.join(dirname, "crystal_ligand.mol2")
    ref_mol = Chem.MolFromMol2File(ref_path)
    if ref_mol is None:
        print("[WARN] Mol2 load failed, trying PDB")
        ref_path = os.path.join(dirname, "crystal_ligand.pdb")
        if os.path.exists(ref_path):
            ref_mol = Chem.MolFromPDBFile(ref_path)
    
    if ref_mol:
        print("[OK] Reference loaded: %d atoms" % ref_mol.GetNumAtoms())
    else:
        print("[WARN] No reference loaded")
except Exception as e:
    print("[FAIL] %s" % e)
    import traceback
    traceback.print_exc()

# Test 4: Pharmacophore features
print("\n=== Test 4: Pharmacophore Features ===")
try:
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    
    feat_file = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feat_factory = ChemicalFeatures.BuildFeatureFactory(feat_file)
    feats = feat_factory.GetFeaturesForMol(mol)
    
    print("[OK] Found %d features" % len(feats))
    for feat in feats:
        pos = feat.GetPos()
        print("  - %s at (%.2f, %.2f, %.2f)" % (
            feat.GetFamily(), pos.x, pos.y, pos.z
        ))
except Exception as e:
    print("[FAIL] %s" % e)
    import traceback
    traceback.print_exc()

# Test 5: Feature matrix generation (simplified)
print("\n=== Test 5: Feature Matrix Generation ===")
try:
    feats_dic = {"Donor":0,"Acceptor":1,"PosIonizable":2,"Aromatic":3,"Hydrophobe":4,"LumpedHydrophobe":5}
    
    # Create simple grid
    STEP = 1.0
    min_x, max_x = -15, 20
    min_y, max_y = -20, 18
    min_z, max_z = -5, 48
    
    xxx,yyy,zzz = np.mgrid[min_x:max_x:STEP,min_y:max_y:STEP,min_z:max_z:STEP]
    grid_shape = xxx.shape
    print("[OK] Grid shape: %s" % str(grid_shape))
    
    # Map features to grid
    feat_list = []
    for feat in feats:
        pos = feat.GetPos()
        x_grid = int((pos.x - min_x) / STEP)
        y_grid = int((pos.y - min_y) / STEP)
        z_grid = int((pos.z - min_z) / STEP)
        feat_type = feats_dic.get(feat.GetFamily(), -1)
        if feat_type >= 0:
            feat_list.append({
                'family': feat.GetFamily(),
                'x': pos.x, 'y': pos.y, 'z': pos.z,
                'x_grid': x_grid, 'y_grid': y_grid, 'z_grid': z_grid,
                'feat_type': feat_type
            })
    
    print("[OK] Mapped %d features to grid" % len(feat_list))
    
    # Create feature matrix (simplified - no pocket mask)
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    feat_matrix = np.zeros((total_grids, len(feats_dic)))
    
    for f in feat_list:
        flat_idx = np.ravel_multi_index(
            [f['x_grid'], f['y_grid'], f['z_grid']], 
            grid_shape
        )
        if flat_idx < total_grids:
            feat_matrix[flat_idx, f['feat_type']] = 1
    
    print("[OK] Feature matrix shape: %s" % str(feat_matrix.shape))
    
    # Save as tensor
    import pickle
    tensor_path = os.path.join(dirname, "tensor_rdkit_only.pkl")
    with open(tensor_path, 'wb') as f:
        pickle.dump([feat_matrix.flatten()], f)
    print("[OK] Saved tensor to %s" % tensor_path)
    
except Exception as e:
    print("[FAIL] %s" % e)
    import traceback
    traceback.print_exc()

print("\n=== RDKit-only pipeline complete ===")
