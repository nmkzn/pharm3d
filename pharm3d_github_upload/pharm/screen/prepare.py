import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter, AllChem
import concurrent, os, pickle
import concurrent.futures
from functools import partial

from .utils import genConfs,gen3D,alignMol,featurizer,write_log


def box_info(box_path):
    with open(box_path) as file:
        data_box=[int(i) for i in file.read().split()]
    min_x = data_box[0]
    max_x = data_box[1]
    min_y = data_box[2]
    max_y = data_box[3]
    min_z = data_box[4]
    max_z = data_box[5]
    return min_x, max_x, min_y, max_y, min_z, max_z

def screen_read_mols(dirname, num_confs=1, ncpu=5):
    min_x, max_x, min_y, max_y, min_z, max_z = box_info(os.path.join(dirname, "box_info.txt"))

    smiles = pd.read_csv(os.path.join(dirname, "template_screen_mols.smi"), sep=',', names=['smiles', 'ChemBL'], usecols=[0,1], header=None)
    df = [[0] for _ in range(len(smiles['smiles'].tolist()))]
    for i in range(len(smiles['smiles'].tolist())):
        df[i] = pd.DataFrame({'smiles': [smiles.iloc[i,0]] * num_confs,
                            'ChemBL': [smiles.iloc[i,1]] * num_confs,
                            })
    df = pd.concat(df,ignore_index=True)
    sdwriter = SDWriter(os.path.join(dirname, "template_screen_mols.sdf"))
    conf2D = [[[0] for _ in range(int(num_confs))] for _ in range(len(smiles['smiles']))]
    mols = [[[0] for _ in range(int(num_confs))] for _ in range(len(smiles['smiles']))]
    for i in smiles.index:
        row = smiles.loc[i]
        try:
            genConfs(row=row,num_confs=num_confs, sdwriter=sdwriter, conformers=conf2D, mols=mols, ncpu=ncpu)
        except Exception as e:
            write_log(dirname, str(e))
    sdwriter.close()

    ## clean df
    nan_indexes = []
    for i in range(len(smiles)):
        for idx, item in enumerate(conf2D[i]):
            if item == [0]:
                nan_indexes.append((i, idx))
                conf2D[i][idx] = np.nan
                mols[i][idx] = np.nan
    df['mols'] = np.hstack(mols)
    df = df.dropna()

    ## embed 3d
    write_log(dirname, "Embed 3d ......")
    df['confs_obj'] = df['mols'].apply(lambda mol: gen3D(mol))
    df = df.dropna()
    df.reset_index(drop=True,inplace=True)
    write_log(dirname, f"3D embedded number:{len(df['confs_obj'].tolist())}")

    ## align
    write_log(dirname, "Start alignment ......")
    reference_path = os.path.join(dirname, "crystal_ligand.mol2")
    tmol = Chem.MolFromMol2File(reference_path)
    df['nmols'] = df['confs_obj'].apply(lambda x: x.mol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
        partial_align = partial(alignMol, ref=tmol)
        results = list(executor.map(partial_align, [df.loc[i] for i in df.index]))
    df['rmsd'] = results
    df = df.dropna()
    df.reset_index(drop=False,inplace=True)
    write_log(dirname, f"aligned rmsd number:{len(df['rmsd'].tolist())}")

    ## featurize with fixed box
    write_log(dirname, "Featurize with fixed box ......")
    with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as executor:
        partial_featurizer = partial(featurizer,min_x=min_x,min_y=min_y,min_z=min_z,max_x=max_x,max_y=max_y,max_z=max_z)
        molfeat = list(executor.map(partial_featurizer, [df.loc[i] for i in df.index]))
    df['moleculeFeaturizer'] = molfeat
    df = df.dropna()
    df.reset_index(drop=True,inplace=True)
    df.to_csv(os.path.join(dirname, "test.csv"), index=False)
    featDf = df['moleculeFeaturizer'].apply(lambda x: x.newdf).tolist()
    pickle.dump(featDf, open(os.path.join(dirname, "test.pkl"),'wb'))

    # save aligned sdf as the last step since some aligned molecules cannot be embedded into the specific grid
    write_log(dirname, "Save aligned sdf as the last step since some aligned molecules cannot be embedded into the specific grid ......")
    sdwriter = SDWriter(os.path.join(dirname, "test.sdf"))
    n = 0
    for chembl, mol in zip(df['ChemBL'].tolist(),df['nmols'].tolist()):
        mol.SetProp('_Name', f"{chembl}_{n}")
        sdwriter.write(mol)
        n += 1
    sdwriter.close()
    write_log(dirname, f"aligned conformer number:{len(df['nmols'].tolist())}")

def screen_match(dirname, weight, ncpu=5, nfeat=4):
    write_log(dirname, 'Start screening ......')
    dataset_pkl = os.path.join(dirname, "test.pkl")
    dataset_sdf = os.path.join(dirname, "test.sdf")
    model_weight = os.path.join(dirname, weight)
    matched_txt = os.path.join(dirname, 'matched_molIdx.txt')
    matched_sdf = os.path.join(dirname, 'matched_molIdx.sdf')

    ## load trained attention weight matrix
    write_log(dirname, "load trained attention weight matrix")
    indicesAttWtPd = pd.read_csv(model_weight)
    indicesAttWtPdSort = indicesAttWtPd.sort_values(by=['weight'],ascending=False)
    indicesAttWtPdSort = indicesAttWtPdSort.reset_index(drop=True)

    ## load screened dataset (only feature df with identical box size)
    fda_feat_data = pickle.load(open(dataset_pkl,'rb'))
    write_log(dirname, f"screened dataset:{len(fda_feat_data)}")

    ## construct 4-dimensional arrays with weight (match feature type & coordinate grid points)
    df = [pd.DataFrame(data=mol, columns=['family','x','y','z','x_grid_No','y_grid_No','z_grid_No','featNo','flatGridNo']) for mol in fda_feat_data]
    screen_mols = [df_i[['featNo','x_grid_No','y_grid_No','z_grid_No']].values for df_i in df]
    weight_feats = indicesAttWtPdSort[:20][['featNo','grid_x','grid_y','grid_z','weight']].values

    write_log(dirname, "Find out molecules that match pharmacophore.")
    ## find out molecules that match pharmacophore
    molsID = []   ## 1D molecule indices in the total screened sdf (shape: (Num. tot. matched mol., ))
    featsID = []   ## feature indices used below
    for i, mol_feats in enumerate(screen_mols):
        matched_per_mol = []
        matched_feat = []
        for j, single_feat in enumerate(mol_feats):
            matched_per_mol.append(np.all((np.array(mol_feats)[:,None]==np.array(weight_feats)[:,:4])[j],axis=1))
            matched_feat.append(np.argwhere(np.all(np.array((mol_feats)[:,None]==np.array(weight_feats)[:,:4])[j],axis=1)))
        if np.sum(np.hstack(matched_per_mol)) >= int(nfeat):
            molsID.append(i+1)
            featsID.append(matched_feat)
    np.savetxt(matched_txt,list(enumerate(molsID)),fmt='%s')
    write_log(dirname, "Save matched molecule index to txt file done.")

    ## write matched molecules as sdf
    aligned_mol = AllChem.SDMolSupplier(dataset_sdf)
    mol_list = []
    for mol in aligned_mol:
        mol_list.append(mol)
    write_log(dirname, f"Matched mols:{len(mol_list)}")
    sdwriter = AllChem.SDWriter(matched_sdf)
    for idx, i in enumerate(molsID):
        mol = aligned_mol[i]
        if mol is not None:
            mol.SetProp('_Name',f"{i}_{idx+1}")
            sdwriter.write(mol)
    sdwriter.close()
    write_log(dirname, "Write matched molecules as sdf down.")
    write_log(dirname, "Screening finished ......")
