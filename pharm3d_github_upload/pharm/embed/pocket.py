import numpy as np
import pandas as pd
from rdkit import Chem
import multiprocessing as mp
from rdkit.Chem import SDWriter
from pymol import cmd
import concurrent, gc, os, pickle
from functools import partial

from .utils import pocketGrids, genConfs, gen3D, alignMol, genBoundary, write_log, featurizer, processMatrix, getSimplifiedMatrix, partial_multiply

def get_protein_ligand_neighbors(dirname,ligand_residue_id='UNK',cutoff_distance=5):
    pdb = os.path.join(dirname, "template_complex.pdb")
    cmd.load(pdb,'protein_ligand_complex')
    cmd.select('ligand',f'resn {ligand_residue_id}')
    cmd.select('protein',f'not resn {ligand_residue_id}')
    ligand_coords = cmd.get_coords('ligand')
    protein_coords = cmd.get_coords('protein')
    ligand_atom_names = []
    protein_atom_names = []
    cmd.iterate('ligand', 'ligand_atom_names.append((chain, resi, name))', space=locals())
    cmd.iterate('protein', 'protein_atom_names.append((chain, resi, name))', space=locals())
    valid_distances = []
    for ligand_atom_name in ligand_atom_names:
        for protein_atom_name in protein_atom_names:
            distance_value = cmd.get_distance(atom1=f"{ligand_atom_name[0]}/{ligand_atom_name[1]}/{ligand_atom_name[2]}", \
                    atom2=f"{protein_atom_name[0]}/{protein_atom_name[1]}/{protein_atom_name[2]}")
            if distance_value < cutoff_distance:
                ligand_atom_coord = cmd.get_coords(f'{ligand_atom_name[0]}/{ligand_atom_name[1]}/{ligand_atom_name[2]}')
                protein_atom_coord = cmd.get_coords(f'{protein_atom_name[0]}/{protein_atom_name[1]}/{protein_atom_name[2]}')
                valid_distances.append((ligand_atom_coord,protein_atom_coord))
    path = os.path.join(dirname, f'{cutoff_distance}A_dist_info.npy')
    valid = np.array(valid_distances)
    np.save(path, valid)
    write_log(dirname,"the pocket info is saved into "+ path)
    cmd.delete("all")
    return path

def vertices_gen(pocket_path_npy, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=5):
    dist = np.load(pocket_path_npy,allow_pickle=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
        partial_pocketGrids = partial(pocketGrids,min_x=min_x,min_y=min_y,min_z=min_z,max_x=max_x,max_y=max_y,max_z=max_z)
        vertices = partial_pocketGrids(dist)
    return vertices

def read_smiles(dirname, num_confs=5, ncpu=5):
    smiles_path = os.path.join(dirname, "template_known_mols.smi")
    reference_path = os.path.join(dirname, "crystal_ligand.mol2")
    conformer_path = os.path.join(dirname, "confs.sdf")
    align_path = os.path.join(dirname, "aligned.sdf")
    box_path = os.path.join(dirname, "box_info.txt")
    #read file
    smiles = pd.read_csv(smiles_path, sep=',', names=['smiles', 'ChemBL', 'states'], usecols=[0,1,2], header=None)
    df = [[0] for _ in range(len(smiles['smiles'].tolist()))]
    for i in range(len(smiles['smiles'].tolist())):
        df[i] = pd.DataFrame({'smiles': [smiles.iloc[i,0]] * num_confs,
                            'ChemBL': [smiles.iloc[i,1]] * num_confs,
                            'states': [smiles.iloc[i,2]] * num_confs
                            })
    df = pd.concat(df,ignore_index=True)

    #generate conformers
    sdwriter = SDWriter(conformer_path)
    results = []
    # with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
    #     partial_genConfs = partial(genConfs, num_confs=num_confs, sdwriter=sdwriter)
    #     results = list(executor.map(partial_genConfs, [df.loc[i] for i in df.index]))
    for i in df.index:
        conformer = genConfs(df.loc[i],num_confs,sdwriter,ncpu=ncpu)
        results.append(conformer)
    df['confs'] = results
    sdwriter.close()

    mol_list = []
    supplier = Chem.SDMolSupplier(conformer_path)
    for mol in supplier:
        mol_list.append(mol)
    df['mols'] = pd.DataFrame(data=mol_list)
    del mol_list
    gc.collect()

    #embed 3d
    df['confs_obj'] = df['mols'].apply(lambda mol: gen3D(mol))
    df = df.dropna()
    df = df.reset_index(drop=True)

    ## align
    tmol = Chem.MolFromMol2File(reference_path)
    df['nmols'] = df['confs_obj'].apply(lambda x: x.mol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
        partial_align = partial(alignMol, ref=tmol)
        results = list(executor.map(partial_align, [df.loc[i] for i in df.index]))
    df['rmsd'] = results
    df = df.drop(np.where(pd.isna(df['rmsd']))[0],axis=0)
    df = df.reset_index(drop=True)
    df['nmols_obj'] = df['nmols'].apply(lambda mol: gen3D(mol))
    sdwriter = SDWriter(align_path)
    for chembl, mol in zip(df['ChemBL'].tolist(),df['nmols'].tolist()):
        mol.SetProp('_Name', f"{chembl}")
        sdwriter.write(mol)
    sdwriter.close()

    ## get boundary
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
        results = list(executor.map(genBoundary, [df.loc[i] for i in df.index]))
    df['boundary'] = results
    df['boundary_x_min'] = df['boundary'].apply(lambda x : x[0])
    df['boundary_x_max'] = df['boundary'].apply(lambda x : x[1])
    df['boundary_y_min'] = df['boundary'].apply(lambda x : x[2])
    df['boundary_y_max'] = df['boundary'].apply(lambda x : x[3])
    df['boundary_z_min'] = df['boundary'].apply(lambda x : x[4])
    df['boundary_z_max'] = df['boundary'].apply(lambda x : x[5])
    min_x = df['boundary_x_min'].min()
    max_x = df['boundary_x_max'].max()
    min_y = df['boundary_y_min'].min()
    max_y = df['boundary_y_max'].max()
    min_z = df['boundary_z_min'].min()
    max_z = df['boundary_z_max'].max()
    # print("min_x:{},max_x:{},min_y:{},max_y:{},min_z:{},max_z:{}".format(min_x,max_x,min_y,max_y,min_z,max_z))
    write_log(dirname, "min_x:{},max_x:{},min_y:{},max_y:{},min_z:{},max_z:{}".format(min_x,max_x,min_y,max_y,min_z,max_z))
    df = df.dropna()
    df.reset_index(drop=False, inplace=True)
    with open(box_path,'w') as f:
        f.write("{} {} {} {} {} {}".format(min_x,max_x,min_y,max_y,min_z,max_z))

    return min_x,max_x,min_y,max_y,min_z,max_z,df

def featurizer_new(dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=5):
    ## split dataframe pre 500 records
    idx=[]
    idx_file=[]
    i=0
    while i<df.shape[0]:
        idx.append(i)
        idx_file.append(os.path.join(dirname,"featMatrix_"+str(i)+".pkl"))
        i += 500
    idx.append(df.shape[0])
                        
    ## Declare ind_att_pd
    tmp_molfeatMatrix = featurizer(row=df.loc[0],min_x=min_x,min_y=min_y,min_z=min_z,max_x=max_x,max_y=max_y,max_z=max_z,vertices=vertices)
    ind_att_pd = np.ones(tmp_molfeatMatrix.shape)

    ##    featurizer start
    write_log(dirname, "Featurizer start ......")
    labels = []
    for i in range(1,len(idx)):
        write_log(dirname,"batch_" + str(i))
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
            partial_featurizer = partial(featurizer,min_x=min_x,min_y=min_y,min_z=min_z,max_x=max_x,max_y=max_y,max_z=max_z,vertices=vertices)
            molfeatMatrix = list(executor.map(partial_featurizer, [df.loc[i] for i in range(idx[i-1],idx[i])]))

        molfeatMatrixClean = []
        for Matrix in molfeatMatrix:
            if isinstance(Matrix, np.ndarray):
                molfeatMatrixClean.append(Matrix)
                labels.append(1)
            else:
                labels.append(None)
        del molfeatMatrix
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
            tmp = list(executor.map(processMatrix, molfeatMatrixClean))
            
        ind_att_pd *= partial_multiply(tmp)
        pickle.dump(molfeatMatrixClean, open(idx_file[i-1], 'wb'))
        del molfeatMatrixClean
        del tmp
        gc.collect()
    write_log(dirname, "Featurizer finished ......")

    df["None_label"] = labels
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(dirname,'state.csv'),index=False)

    ## Generate and save indicesAttPd
    write_log(dirname, "Generate and save indicesAttPd ......")
    indicesAtt = np.where(ind_att_pd==0)
    indicesAttPd = pd.DataFrame(data={'gridNo':indicesAtt[0],'featNo':indicesAtt[1]})
    indicesAttPd.to_csv(os.path.join(dirname,'ind_att_pd.csv'),index=False)

    ## Starting molfeatMatrixSimplified generate
    write_log(dirname, "Starting molfeatMatrixSimplified generate ......")
    molfeatMatrixSimplified = []
    for i in range(1,len(idx)):
        molfeatMatrix = pickle.load(open(idx_file[i-1], 'rb'))
        write_log(dirname, idx_file[i-1])
        with mp.Pool(processes=ncpu) as pool:
            partial_simplifier = partial(getSimplifiedMatrix,ind_att_pd=indicesAttPd)
            molfeatMatrixSimplified += list(pool.map(partial_simplifier, molfeatMatrix))
    pickle.dump(molfeatMatrixSimplified,open(os.path.join(dirname,'tensor.pkl'), 'wb'))

    return os.path.join(dirname,'ind_att_pd.csv'), os.path.join(dirname,'tensor.pkl'), os.path.join(dirname,'state.csv')