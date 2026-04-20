"""Single molecule processing - RDKit-only pipeline (no PyMOL)"""
import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from Bio.PDB import PDBParser

sys.path.insert(0, r'E:\pharm3d')

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
log_dir = r'E:\pharm3d\one_molecule_run\logs'
os.makedirs(log_dir, exist_ok=True)

def log(msg):
    print(msg)
    with open(os.path.join(log_dir, 'run_rdkit.log'), 'a') as f:
        f.write(msg + '\n')

def load_protein_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atoms.append(atom.coord)
    return np.array(atoms)

def extract_pocket_rdkit(dirname, cutoff=5):
    """Extract pocket using RDKit instead of PyMOL"""
    pdb_path = os.path.join(dirname, 'template_complex.pdb')
    ref_path = os.path.join(dirname, 'crystal_ligand.mol2')
    
    # Load ligand
    ligand = Chem.MolFromMol2File(ref_path)
    ligand_coords = ligand.GetConformer().GetPositions()
    
    # Load protein
    protein_coords = load_protein_pdb(pdb_path)
    
    # Find pocket atoms
    pocket_coords = []
    for prot_coord in protein_coords:
        distances = np.sqrt(np.sum((ligand_coords - prot_coord)**2, axis=1))
        if np.min(distances) < cutoff:
            pocket_coords.append(prot_coord)
    
    # Save in compatible format
    pocket_path = os.path.join(dirname, '5A_dist_info.npy')
    valid_distances = []
    for lig_coord in ligand_coords[:10]:
        for pocket_coord in pocket_coords[:100]:
            valid_distances.append((lig_coord, pocket_coord))
    np.save(pocket_path, np.array(valid_distances))
    
    return pocket_path, np.array(pocket_coords)

def read_smiles_rdkit(dirname):
    """Read SMILES and generate 3D"""
    smiles_path = os.path.join(dirname, 'template_known_mols.smi')
    df_smi = pd.read_csv(smiles_path, header=None, names=['smiles', 'id', 'label'])
    
    molecules = []
    for _, row in df_smi.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            molecules.append({
                'nmols': mol,
                'index': row['id'],
                'label': row['label']
            })
    
    # Get bounds from box_info
    with open(os.path.join(dirname, 'box_info.txt'), 'r') as f:
        bounds = [int(x) for x in f.read().strip().split()]
    
    return bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], pd.DataFrame(molecules)

def featurize_molecules(df, pocket_coords, box_bounds):
    """Generate feature matrices"""
    min_x, max_x, min_y, max_y, min_z, max_z = box_bounds
    grid_shape = (max_x - min_x, max_y - min_y, max_z - min_z)
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    feats_dic = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
                 "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}
    
    feat_file = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feat_factory = ChemicalFeatures.BuildFeatureFactory(feat_file)
    
    tensors = []
    state_records = []
    
    for idx, row in df.iterrows():
        mol = row['nmols']
        feats = feat_factory.GetFeaturesForMol(mol)
        
        feat_matrix = np.zeros((total_grids, len(feats_dic)))
        for feat in feats:
            pos = feat.GetPos()
            x_grid = int((pos.x - min_x))
            y_grid = int((pos.y - min_y))
            z_grid = int((pos.z - min_z))
            
            if 0 <= x_grid < grid_shape[0] and 0 <= y_grid < grid_shape[1] and 0 <= z_grid < grid_shape[2]:
                flat_idx = np.ravel_multi_index([x_grid, y_grid, z_grid], grid_shape)
                feat_type = feats_dic.get(feat.GetFamily(), -1)
                if feat_type >= 0:
                    feat_matrix[flat_idx, feat_type] = 1
        
        tensors.append(feat_matrix.flatten())
        state_records.append({
            'ID': row['index'],
            'label': row['label'],
            'active_conf': 0,
            'total_conf': 1
        })
    
    return tensors, pd.DataFrame(state_records)

def main():
    log("=== RDKit-Only Pipeline Start ===\n")
    total_t0 = time.time()
    
    # Step 1: Pocket extraction (RDKit)
    log("Step 1: Pocket Extraction (RDKit)")
    t0 = time.time()
    pocket_path, pocket_coords = extract_pocket_rdkit(dirname, cutoff=5)
    log("[OK] Pocket: %d atoms" % len(pocket_coords))
    log("  Time: %.2fs" % (time.time() - t0))
    
    # Step 2: Read SMILES
    log("\nStep 2: Read SMILES")
    t0 = time.time()
    min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles_rdkit(dirname)
    log("[OK] Molecules: %d" % len(df))
    log("  Time: %.2fs" % (time.time() - t0))
    log("  Bounds: x[%d,%d] y[%d,%d] z[%d,%d]" % (min_x, max_x, min_y, max_y, min_z, max_z))
    
    # Step 3: Featurization
    log("\nStep 3: Featurization")
    t0 = time.time()
    tensors, state_df = featurize_molecules(df, pocket_coords, (min_x, max_x, min_y, max_y, min_z, max_z))
    log("[OK] Tensors: %d" % len(tensors))
    log("  Time: %.2fs" % (time.time() - t0))
    log("  Feature dim: %d" % len(tensors[0]))
    
    # Step 4: Save outputs
    log("\nStep 4: Save Outputs")
    
    tensor_path = os.path.join(dirname, "tensor.pkl")
    with open(tensor_path, 'wb') as f:
        pickle.dump(tensors, f)
    log("[OK] tensor.pkl: %d bytes" % os.path.getsize(tensor_path))
    
    state_path = os.path.join(dirname, "state.csv")
    state_df.to_csv(state_path, index=False)
    log("[OK] state.csv: %d bytes" % os.path.getsize(state_path))
    
    # ind_att_pd.csv
    feats_dic = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
                 "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}
    ind_att_pd = pd.DataFrame([
        {'feature': k, 'index': v} for k, v in feats_dic.items()
    ])
    ind_att_pd_path = os.path.join(dirname, "ind_att_pd.csv")
    ind_att_pd.to_csv(ind_att_pd_path, index=False)
    log("[OK] ind_att_pd.csv: %d bytes" % os.path.getsize(ind_att_pd_path))
    
    total_time = time.time() - total_t0
    log("\n=== Pipeline Complete ===")
    log("Total time: %.2fs" % total_time)
    log("Output: %s" % tensor_path)
    
    return tensor_path

if __name__ == "__main__":
    main()
