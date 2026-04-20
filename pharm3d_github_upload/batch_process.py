# -*- coding: utf-8 -*-
"""
Batch processing script for multiple molecules
Usage: python batch_process.py <input_smi_file> <output_dir> [num_confs]
"""
import sys
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from Bio.PDB import PDBParser

# Fixed training box bounds
BOX_BOUNDS = (-14, 19, -19, 17, 0, 47)
FEATS_DIC = {"Donor":0, "Acceptor":1, "PosIonizable":2, 
             "Aromatic":3, "Hydrophobe":4, "LumpedHydrophobe":5}

def load_protein_pdb(pdb_path):
    """Load protein structure from PDB"""
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
                        'coord': atom.coord
                    })
    return pd.DataFrame(atoms)

def extract_pocket(protein_pdb, ligand_mol, cutoff=5):
    """Extract pocket atoms within cutoff distance of ligand"""
    ligand_coords = ligand_mol.GetConformer().GetPositions()
    protein_df = load_protein_pdb(protein_pdb)
    
    pocket_coords = []
    for _, prot_atom in protein_df.iterrows():
        distances = np.sqrt(np.sum((ligand_coords - prot_atom['coord'])**2, axis=1))
        if np.min(distances) < cutoff:
            pocket_coords.append(prot_atom['coord'])
    
    return np.array(pocket_coords)

def process_molecule(smiles, mol_id, ref_mol, pocket_coords, feat_factory):
    """Process single molecule through pipeline"""
    # Generate 3D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Align to reference
    try:
        from rdkit.Chem import rdMolAlign
        rdMolAlign.AlignMol(mol, ref_mol)
    except:
        pass  # Continue even if alignment fails
    
    # Featurize
    min_x, max_x, min_y, max_y, min_z, max_z = BOX_BOUNDS
    grid_shape = (max_x - min_x, max_y - min_y, max_z - min_z)
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    feats = feat_factory.GetFeaturesForMol(mol)
    feat_matrix = np.zeros((total_grids, len(FEATS_DIC)))
    
    for feat in feats:
        pos = feat.GetPos()
        x_grid = int((pos.x - min_x))
        y_grid = int((pos.y - min_y))
        z_grid = int((pos.z - min_z))
        
        if 0 <= x_grid < grid_shape[0] and 0 <= y_grid < grid_shape[1] and 0 <= z_grid < grid_shape[2]:
            flat_idx = np.ravel_multi_index([x_grid, y_grid, z_grid], grid_shape)
            feat_type = FEATS_DIC.get(feat.GetFamily(), -1)
            if feat_type >= 0:
                feat_matrix[flat_idx, feat_type] = 1
    
    return feat_matrix.flatten(), None

def reflatten_with_index(tensor, fixed_index):
    """Re-flatten tensor using fixed training index"""
    min_x, max_x, min_y, max_y, min_z, max_z = BOX_BOUNDS
    grid_shape = (max_x - min_x, max_y - min_y, max_z - min_z)
    total_grids = grid_shape[0] * grid_shape[1] * grid_shape[2]
    
    grid_features = tensor.reshape(total_grids, len(FEATS_DIC))
    
    flattened = []
    for _, row in fixed_index.iterrows():
        grid_no = row['gridNo']
        feat_no = row['featNo']
        if grid_no < total_grids and feat_no < len(FEATS_DIC):
            flattened.append(grid_features[grid_no, feat_no])
        else:
            flattened.append(0.0)
    
    return np.array(flattened)

def main():
    parser = argparse.ArgumentParser(description='Batch process molecules')
    parser.add_argument('input_smi', help='Input SMILES file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--ref', default=r'E:\pharm3d\static\files\crystal_ligand.mol2',
                        help='Reference ligand for alignment')
    parser.add_argument('--protein', default=r'E:\pharm3d\static\files\template_complex.pdb',
                        help='Protein PDB for pocket extraction')
    parser.add_argument('--index', default=r'E:\pharm3d\00011734250639\ind_att_pd.csv',
                        help='Fixed training index')
    parser.add_argument('--cutoff', type=float, default=5, help='Pocket cutoff distance')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Batch Molecule Processing")
    print("=" * 60)
    
    # Load reference and pocket
    print("\n[1/4] Loading reference structures...")
    ref_mol = Chem.MolFromMol2File(args.ref)
    if ref_mol is None:
        print("ERROR: Failed to load reference")
        return
    print("  Reference: %d atoms" % ref_mol.GetNumAtoms())
    
    pocket_coords = extract_pocket(args.protein, ref_mol, args.cutoff)
    print("  Pocket: %d atoms (%.1fA cutoff)" % (len(pocket_coords), args.cutoff))
    
    # Load fixed index
    print("\n[2/4] Loading fixed training index...")
    fixed_index = pd.read_csv(args.index)
    print("  Index entries: %d" % len(fixed_index))
    
    # Setup feature factory
    feat_file = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    feat_factory = ChemicalFeatures.BuildFeatureFactory(feat_file)
    
    # Load input molecules
    print("\n[3/4] Processing molecules...")
    input_df = pd.read_csv(args.input_smi, header=None, names=['smiles', 'id', 'label'])
    print("  Input molecules: %d" % len(input_df))
    
    # Process each molecule
    tensors = []
    state_records = []
    errors = []
    
    t0 = time.time()
    for idx, row in input_df.iterrows():
        mol_t0 = time.time()
        tensor, error = process_molecule(
            row['smiles'], row['id'], ref_mol, pocket_coords, feat_factory
        )
        
        if error:
            errors.append((row['id'], error))
            continue
        
        # Re-flatten with fixed index
        tensor_fixed = reflatten_with_index(tensor, fixed_index)
        tensors.append(tensor_fixed)
        
        state_records.append({
            'ID': row['id'],
            'label': row['label'],
            'active_conf': 0,
            'total_conf': 1
        })
        
        if (idx + 1) % 10 == 0 or idx == len(input_df) - 1:
            print("  Processed: %d/%d (%.2fs)" % (idx + 1, len(input_df), time.time() - t0))
    
    process_time = time.time() - t0
    
    # Save outputs
    print("\n[4/4] Saving outputs...")
    
    # tensor.pkl
    tensor_path = os.path.join(args.output_dir, "tensor.pkl")
    with open(tensor_path, 'wb') as f:
        pickle.dump(tensors, f)
    print("  tensor.pkl: %d molecules, %d bytes" % (len(tensors), os.path.getsize(tensor_path)))
    
    # state.csv
    state_path = os.path.join(args.output_dir, "state.csv")
    pd.DataFrame(state_records).to_csv(state_path, index=False)
    print("  state.csv: %d bytes" % os.path.getsize(state_path))
    
    # ind_att_pd.csv (copy of fixed index)
    index_path = os.path.join(args.output_dir, "ind_att_pd.csv")
    fixed_index.to_csv(index_path, index=False)
    print("  ind_att_pd.csv: %d bytes" % os.path.getsize(index_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("Total molecules: %d" % len(input_df))
    print("Successful: %d" % len(tensors))
    print("Failed: %d" % len(errors))
    print("Total time: %.2fs" % process_time)
    print("Avg time/molecule: %.3fs" % (process_time / len(input_df)))
    print("Output directory: %s" % args.output_dir)
    
    if errors:
        print("\nErrors:")
        for mol_id, err in errors[:5]:
            print("  %s: %s" % (mol_id, err))

if __name__ == "__main__":
    main()
