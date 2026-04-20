# -*- coding: utf-8 -*-
"""
Complete RDKit-based pipeline for single molecule processing
Generates tensor.pkl compatible with trained MLP models
"""
import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

sys.path.insert(0, r'E:\pharm3d')

def load_protein_pdb(pdb_path):
    """Load protein structure from PDB"""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]
                res_name = residue.get_resname()
                for atom in residue:
                    atoms.append({
                        'chain': chain.id,
                        'resi': res_id,
                        'resn': res_name,
                        'name': atom.name,
                        'coord': atom.coord,
                        'element': atom.element
                    })
    return pd.DataFrame(atoms)

def extract_pocket_rdkit(dirname, cutoff_distance=5):
    """Extract pocket atoms using RDKit/Biopython instead of PyMOL"""
    pdb_path = os.path.join(dirname, "template_complex.pdb")
    
    # Load ligand from reference
    ref_path = os.path.join(dirname, "crystal_ligand.mol2")
    ligand = Chem.MolFromMol2File(ref_path)
    if ligand is None:
        raise ValueError("Failed to load reference ligand")
    
    # Get ligand coordinates
    ligand_conf = ligand.GetConformer()
    ligand_coords = ligand_conf.GetPositions()
    
    # Load protein
    protein_df = load_protein_pdb(pdb_path)
    
    # Find protein atoms within cutoff distance
    pocket_atoms = []
    for _, prot_atom in protein_df.iterrows():
        prot_coord = prot_atom['coord']
        # Check distance to all ligand atoms
        distances = np.sqrt(np.sum((ligand_coords - prot_coord)**2, axis=1))
        if np.min(distances) < cutoff_distance:
            pocket_atoms.append(prot_coord)
    
    # Save pocket info
    pocket_path = os.path.join(dirname, f'{cutoff_distance}A_dist_info.npy')
    
    # Create distance pairs (ligand_atom, protein_atom) for compatibility
    valid_distances = []
    for lig_coord in ligand_coords[:10]:  # Sample first 10 ligand atoms
        for pocket_coord in pocket_atoms[:100]:  # Limit for efficiency
            valid_distances.append((lig_coord, pocket_coord))
    
    np.save(pocket_path, np.array(valid_distances))
    print("  Pocket saved: %s (%d pairs)" % (pocket_path, len(valid_distances)))
    
    return pocket_path, pocket_atoms

def generate_conformers(dirname, num_confs=5):
    """Generate 3D conformers from SMILES"""
    smiles_path = os.path.join(dirname, "template_known_mols.smi")
    df = pd.read_csv(smiles_path, header=None, names=['smiles', 'id', 'label'])
    
    molecules = []
    for _, row in df.iterrows():
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
    
    return pd.DataFrame(molecules)

def align_to_reference(dirname, mol_df):
    """Align molecules to reference crystal ligand"""
    ref_path = os.path.join(dirname, "crystal_ligand.mol2")
    ref_mol = Chem.MolFromMol2File(ref_path)
    
    if ref_mol is None or len(mol_df) == 0:
        return mol_df
    
    from rdkit.Chem import rdMolAlign
    
    aligned = []
    for _, row in mol_df.iterrows():
        mol = row['nmols']
        try:
            # Try alignment
            rmsd = rdMolAlign.AlignMol(mol, ref_mol)
            row['rmsd'] = rmsd
        except:
            row['rmsd'] = None
        aligned.append(row)
    
    return pd.DataFrame(aligned)

def featurize_molecules(mol_df, pocket_atoms, box_bounds, step=1.0):
    """Generate feature matrices for molecules"""
    min_x, max_x, min_y, max_y, min_z, max_z = box_bounds
    
    # Create grid
    xxx, yyy, zzz = np.mgrid[min_x:max_x:step, min_y:max_y:step, min_z:max_z:step]
    grid_shape = xxx.shape
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    feats_dic = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
                 "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}
    
    feat_file = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feat_factory = ChemicalFeatures.BuildFeatureFactory(feat_file)
    
    tensors = []
    state_records = []
    
    for idx, row in mol_df.iterrows():
        mol = row['nmols']
        mol_id = row['index']
        label = row['label']
        
        # Get features
        feats = feat_factory.GetFeaturesForMol(mol)
        
        # Create feature matrix
        feat_matrix = np.zeros((total_grids, len(feats_dic)))
        
        for feat in feats:
            pos = feat.GetPos()
            x_grid = int((pos.x - min_x) / step)
            y_grid = int((pos.y - min_y) / step)
            z_grid = int((pos.z - min_z) / step)
            
            # Check bounds
            if 0 <= x_grid < grid_shape[0] and 0 <= y_grid < grid_shape[1] and 0 <= z_grid < grid_shape[2]:
                flat_idx = np.ravel_multi_index([x_grid, y_grid, z_grid], grid_shape)
                feat_type = feats_dic.get(feat.GetFamily(), -1)
                if feat_type >= 0:
                    feat_matrix[flat_idx, feat_type] = 1
        
        tensors.append(feat_matrix.flatten())
        state_records.append({
            'ID': mol_id,
            'label': label,
            'active_conf': 0,
            'total_conf': 1
        })
    
    return tensors, pd.DataFrame(state_records), grid_shape

def main():
    dirname = r'E:\pharm3d\one_molecule_run\job_0001'
    
    print("=== RDKit Pipeline Start ===\n")
    total_t0 = time.time()
    
    # Step 1: Pocket extraction
    print("Step 1: Pocket Extraction")
    t0 = time.time()
    try:
        pocket_path, pocket_atoms = extract_pocket_rdkit(dirname, cutoff_distance=5)
        print("  Time: %.2fs" % (time.time() - t0))
        print("  Pocket atoms: %d" % len(pocket_atoms))
    except Exception as e:
        print("  Error: %s" % e)
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Generate conformers
    print("\nStep 2: Generate Conformers")
    t0 = time.time()
    mol_df = generate_conformers(dirname, num_confs=5)
    print("  Time: %.2fs" % (time.time() - t0))
    print("  Molecules: %d" % len(mol_df))
    
    # Step 3: Align to reference
    print("\nStep 3: Align to Reference")
    t0 = time.time()
    mol_df = align_to_reference(dirname, mol_df)
    print("  Time: %.2fs" % (time.time() - t0))
    
    # Step 4: Featurization
    print("\nStep 4: Featurization")
    t0 = time.time()
    
    # Use box_info.txt bounds
    box_bounds = (-14, 19, -19, 17, 0, 47)
    tensors, state_df, grid_shape = featurize_molecules(mol_df, pocket_atoms, box_bounds)
    
    print("  Time: %.2fs" % (time.time() - t0))
    print("  Grid shape: %s" % str(grid_shape))
    print("  Feature dim: %d" % len(tensors[0]))
    
    # Step 5: Save outputs
    print("\nStep 5: Save Outputs")
    
    # tensor.pkl
    tensor_path = os.path.join(dirname, "tensor.pkl")
    with open(tensor_path, 'wb') as f:
        pickle.dump(tensors, f)
    print("  tensor.pkl: %d bytes" % os.path.getsize(tensor_path))
    
    # state.csv
    state_path = os.path.join(dirname, "state.csv")
    state_df.to_csv(state_path, index=False)
    print("  state.csv: %d bytes" % os.path.getsize(state_path))
    
    # ind_att_pd.csv (feature index)
    feats_dic = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
                 "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}
    ind_att_pd = pd.DataFrame([
        {'feature': k, 'index': v} for k, v in feats_dic.items()
    ])
    ind_att_pd_path = os.path.join(dirname, "ind_att_pd.csv")
    ind_att_pd.to_csv(ind_att_pd_path, index=False)
    print("  ind_att_pd.csv: %d bytes" % os.path.getsize(ind_att_pd_path))
    
    total_time = time.time() - total_t0
    print("\n=== Pipeline Complete ===")
    print("Total time: %.2fs" % total_time)
    print("Output files in: %s" % dirname)

if __name__ == "__main__":
    main()
