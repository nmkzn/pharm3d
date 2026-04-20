# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')
from embed.pocket import get_protein_ligand_neighbors
import os
dirname = r'E:\pharm3d\one_molecule_run\job_0001'
print("Starting pocket extraction...")
pocket_path = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
print('Pocket generated:', pocket_path)
print('File exists:', os.path.exists(pocket_path))