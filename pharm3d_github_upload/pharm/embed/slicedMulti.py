# by hailu
import os
from rdkit import Geometry
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore
import pandas as pd
import numpy as np
import math
from scipy.spatial import ConvexHull

FEAT = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
featfact = ChemicalFeatures.BuildFeatureFactory(FEAT)
feats_dic = {"Donor":0,"Acceptor":1,"PosIonizable":2,"Aromatic":3,"Hydrophobe":4,"LumpedHydrophobe":5}
class Molecule23D():
    def __init__(self,mol):
        #AllChem.EmbedMolecule(mol)
        #AllChem.MMFFOptimizeMolecule(mol)
        self.mol = mol
        self.molp = AllChem.MMFFGetMoleculeProperties(self.mol)
    def coords2pd(self):
        atoms = self.mol.GetAtoms()
        conformer = self.mol.GetConformer()
        atom_infos = []
        for atom in atoms:
            atom_id = atom.GetIdx()
            atom_pos = conformer.GetAtomPosition(atom_id)
            atom_symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            x = atom_pos.x
            y = atom_pos.y
            z = atom_pos.z
            atom_info = [atom_id,atom_symbol,atomic_num,x,y,z]
            atom_infos.append(atom_info)
        self.df = pd.DataFrame(data=atom_infos,columns=["atom_id","atom_symbol","atomic_num","x","y","z"])
        return self.df         
    def getBoxBoundary(self):
        min_x = math.floor(self.df["x"].min())
        max_x = math.ceil(self.df["x"].max())
        min_y = math.floor(self.df["y"].min())
        max_y = math.ceil(self.df["y"].max())
        min_z = math.floor(self.df["z"].min())
        max_z = math.ceil(self.df["z"].max())
        return [min_x,max_x,min_y,max_y,min_z,max_z]   
    
class MoleculeFeaturizer():
    def __init__(self,df,min_x,max_x,min_y,max_y,min_z,max_z,vertices,STEP=1):
        self.df = df
        self.mol = self.df['nmols']
        self.molID = self.df['index']
        self.feats = featfact.GetFeaturesForMol(self.df['nmols'])
        self.feats4pd = []
        for feat in self.feats:
            pos = feat.GetPos()
            feat4pd = [feat.GetFamily(),pos.x,pos.y,pos.z]
            self.feats4pd.append(feat4pd)
        self.newdf = pd.DataFrame(data=self.feats4pd,columns=["family","x","y","z"])
        xxx,yyy,zzz = np.mgrid[min_x:max_x:STEP,min_y:max_y:STEP,min_z:max_z:STEP]
        self.x_grid_num,self.y_grid_num,self.z_grid_num = xxx.shape
        self.newdf["x_grid_No"],self.newdf["y_grid_No"],self.newdf["z_grid_No"] = zip(*self.newdf.apply(self.getGridNo,args=(min_x,min_y,min_z,STEP),axis=1))
        self.newdf["featNo"] = self.newdf["family"].apply(self.toFeatNo,args=(feats_dic,))
        ## process protein pocket - clash grid assigned -1
        self.vertices = vertices #np.loadtxt('vertices.txt')
        for idx, (x,y,z) in enumerate(zip(self.newdf['x_grid_No'],self.newdf['y_grid_No'],self.newdf['z_grid_No'])):
            points = np.array([x,y,z])
            self.points = points
            if self.point_outside_polyhedron(self.points,vertices):
                self.newdf.loc[idx,"featNo"] = -1
        self.newdf = self.newdf[self.newdf['featNo']!=-1]
        self.newdf["flatGridNo"] = self.newdf.apply(self.toFlatGridNo,args=(self.x_grid_num,self.y_grid_num,self.z_grid_num),axis=1)
        self.feat = self.genFeatMatrix(self.newdf,self.x_grid_num,self.y_grid_num,self.z_grid_num)

    # polyhedron functions
    def point_outside_polyhedron(self, point, vertices):
        def ray_intersects_triangle(ray_origin, ray_direction, triangle):
            EPSILON = 1e-9
            vertex0, vertex1, vertex2 = triangle
            edge1 = vertex1 - vertex0
            edge2 = vertex2 - vertex0
            h = np.cross(ray_direction, edge2)
            a = np.dot(edge1, h)
            if -EPSILON < a < EPSILON:
                return False
            f = 1.0 / a
            s = ray_origin - vertex0
            u = f * np.dot(s,h)
            if u < 0.0 or u > 1.0:
                return False
            q = np.cross(s, edge1)
            v = f * np.dot(ray_direction, q)
            if v < 0.0 or u + v > 1.0:
                return False
            t = f * np.dot(edge2, q)
            if t > EPSILON:
                return True
            else:
                return False

        hull = ConvexHull(vertices)
        faces = hull.simplices
        ray_direction = np.array([1.0,0.0,0.0])
        intersections = 0
        for face in faces:
            triangle = vertices[face]
            if ray_intersects_triangle(point, ray_direction, triangle):
                intersections += 1
        return intersections % 2 == 0

    def getGridNo(self,indf,min_x,min_y,min_z,step):
        raw_x = indf["x"]
        raw_y = indf["y"]
        raw_z = indf["z"]
        x_grid_No = math.floor((raw_x-min_x)/step)
        y_grid_No = math.floor((raw_y-min_y)/step)
        z_grid_No = math.floor((raw_z-min_z)/step)
        return x_grid_No,y_grid_No,z_grid_No
    
    def toFeatNo(self,family,feats_dic):
        if(family in feats_dic.keys()):
            return feats_dic[family]
        else:
            return -1
    
    def toFlatGridNo(self,indf,x_grid_num,y_grid_num,z_grid_num):
        x = indf["x_grid_No"]
        y = indf["y_grid_No"]
        z = indf["z_grid_No"]
        arr = np.array([x,y,z])
        onedId = np.ravel_multi_index(arr,[x_grid_num,y_grid_num,z_grid_num])
        return onedId
    
    def genFeatMatrix(self,df,x_grid_num,y_grid_num,z_grid_num):
        def setFeatVal(feat,row,col):
            feat[row,col] = 1
        feat = np.zeros((x_grid_num*y_grid_num*z_grid_num,len(feats_dic)))   ## name feats is not defined
        df.apply(lambda row: setFeatVal(feat,row["flatGridNo"],row["featNo"]),axis=1)
        return feat

    def getFeatMatrix(self):
        return self.feat


