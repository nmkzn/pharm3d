from .slicedMulti import Molecule23D
from .slicedScreen import MoleculeFeaturizer
import numpy as np
import pandas as pd
import math, os
from rdkit import Chem
from rdkit.Chem import SDWriter,AllChem
import multiprocessing as mp
from functools import partial

#smiles = pd.read_csv('smiles.ism', sep=',', names=['smiles', 'ChemBL', 'states'])
#num_confs = 2
def genConfs(row,num_confs,sdwriter,conformers,mols,ncpu):
    global count_confs
    smiles = row['smiles']
    chembl = row['ChemBL']
    idx = row.name
    mol = Chem.MolFromSmiles(smiles)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, numThreads=ncpu, randomSeed=42)
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        mol.SetProp('_Name', f"{chembl}_{cid+1}")
        conformers[idx][cid] = mol.GetConformer(cid)
        mols[idx][cid] = mol
        sdwriter.write(mol, confId=cid)
    #except Exception as e:
    #    with open('confs_error.log','a') as f:
    #        f.write(f"Error generating multiple confs for {count_confs} {row['smiles']}: {e}\n")
    #        #pass
    #    return None

def gen3D(row):
    try:
        results = Molecule23D(row)
        return results
    except Exception as e:
        with open('embed_error.log', 'a') as f:
            f.write(f'Error generating 3D for {row}: {e}\n')
        return None

def alignMol(row, ref):
    mol = row['nmols']
    try:
        rmsd = AllChem.GetO3A(mol,ref,
                             AllChem.MMFFGetMoleculeProperties(mol),
                             AllChem.MMFFGetMoleculeProperties(ref)).Align()
        return rmsd
    except Exception as e:
        # with open('./align.log', 'a') as f:
        #     f.write(f'Error in {row.name}: {e}\n')
        return None

def genBoundary(row):
    df = row['nmols_obj'].coords2pd()
    boundary = row['nmols_obj'].getBoxBoundary()
    return boundary

def featurizer(row,min_x,max_x,min_y,max_y,min_z,max_z):
    try:
        molfeat = MoleculeFeaturizer(row,min_x,max_x,min_y,max_y,min_z,max_z)
        return molfeat
    except Exception as e:
        with open('./featurizer_error.log', 'a') as f:
            f.write(f'Error featurizing for {row}: {e}\n')
        return None

def pocketGrids(dist,min_x,min_y,min_z,max_x,max_y,max_z):
    pro_int_coords = dist[:,1]
    pro_int_coords = pro_int_coords.tolist()
    pro_x, pro_y, pro_z = [], [], []
    for i in range(len(pro_int_coords)):
        pro_x.append(pro_int_coords[i][0][0])
        pro_y.append(pro_int_coords[i][0][1])
        pro_z.append(pro_int_coords[i][0][2])
    pocket = pd.DataFrame({'x':pro_x,'y':pro_y,'z':pro_z})
    def getGridNo(indf,min_x,min_y,min_z,step):
        raw_x = indf["x"]
        raw_y = indf["y"]
        raw_z = indf["z"]
        x_grid_No = math.floor((raw_x-min_x)/step)
        y_grid_No = math.floor((raw_y-min_y)/step)
        z_grid_No = math.floor((raw_z-min_z)/step)
        return x_grid_No,y_grid_No,z_grid_No
    STEP = 1
    xxx,yyy,zzz = np.mgrid[min_x:max_x:STEP,min_y:max_y:STEP,min_z:max_z:STEP]
    pocket["x_grid_No"],pocket["y_grid_No"],pocket["z_grid_No"] = zip(*pocket.apply(getGridNo,args=(min_x,min_y,min_z,STEP),axis=1))
    vertices = []
    for i in range(len(pocket['x'].tolist())):
        vertices.append([pocket['x_grid_No'].tolist()[i],pocket['y_grid_No'].tolist()[i],pocket['z_grid_No'].tolist()[i]])
    object_vertices = np.array(vertices)
    return object_vertices

def processMatrix(matrix):
    matrix = np.array(matrix).astype(bool)
    matrix = matrix.astype(bool)
    vmat = ~matrix
    vmat = vmat.astype(int)
    return vmat

def partial_multiply(matrixs):
    results = np.ones(matrixs[0].shape)
    for matrix in matrixs:
        results *= matrix
    return results

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def batchGenerator(array, batch_size):
    for i in range(0,len(array),batch_size):
        yield array[i:i+batch_size]

def combineAttention(featMatrixs):
    #temp = np.ones(featMatrixs[0].shape)    
    #for matrix in featMatrixs:
    #    temp *= matrix 
    #return temp
    matrix_chunks = split_list(featMatrixs, mp.cpu_count())
    with mp.Pool(processes=mp.cpu_count()) as pool:
        partial_results = pool.map(partial_multiply, matrix_chunks)
    final_results = np.ones(featMatrixs[0].shape)
    for part in partial_results:
        final_results *= part
    return final_results
    
def getSimplifiedMatrix(matrix,ind_att_pd):
    def getValueAtIndice(indf,originalMatrix):
        return indf['gridNo'],indf['featNo'],originalMatrix[indf['gridNo'],indf['featNo']]
    a = pd.DataFrame()
    a['gridNo'],a['featNo'],a['value'] = zip(*ind_att_pd.apply(getValueAtIndice,args=(matrix,),axis=1))
    simplifiedArr = np.array(a['value'].tolist())
    return simplifiedArr

def drawPoint(row,row_index):
    residue = Residue.Residue((' ',row_index, ' '), 'ALA', ' ')
    chain.add(residue)
    atom = Atom.Atom(atom_types[row["featNo"]], [row["abso_x"],row["abso_y"],row["abso_z"]], 1.0, 1.0, ' ', atom_types[row["featNo"]], row_index, ' ')
    color = atom_colors[row["featNo"]]
    # Convert RGB color to a single integer value for the temperature factor
    temp_factor = 100 * (color[0]*255) + 10 * (color[1]*255) + (color[2]*255)
    atom.set_bfactor(temp_factor)
    residue.add(atom)

def restoreAbsoluteCoord(indf,min_x,min_y,min_z,step):
    grid_x = indf['grid_x']
    grid_y = indf['grid_y']
    grid_z = indf['grid_z']
    abso_x = grid_x * step + min_x + (0.5 * step)
    abso_y = grid_y * step + min_y + (0.5 * step)
    abso_z = grid_z * step + min_z + (0.5 * step)
    return abso_x,abso_y,abso_z

def unravel(gridNo,x_grid_num,y_grid_num,z_grid_num):
    grid_x,grid_y,grid_z = np.unravel_index(gridNo,[x_grid_num,y_grid_num,z_grid_num])
    return grid_x,grid_y,grid_z

def write_log(dirname, message):
    with open(os.path.join(dirname, "job.log"), "a") as file:
        file.write(message)
        file.write("\n")