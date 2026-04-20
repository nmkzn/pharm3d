import time
import sys
import os

# Add pharm to path
sys.path.insert(0, r'E:\pharm3d')
sys.path.insert(0, r'E:\pharm3d\pharm')

# Direct import from pocket.py
from embed.pocket import get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen

dirname = r'E:\pharm3d\one_molecule_run\job_0001'
ncpu = 10
num_confs = 5

print(f"=== 单分子处理测试 ===")
print(f"目录: {dirname}")
print(f"构象数: {num_confs}")
print(f"CPU数: {ncpu}")
print()

# 步骤1: 口袋提取
print("[1/4] 开始口袋提取...")
t0 = time.time()
try:
    pocket_path_npy = get_protein_ligand_neighbors(dirname, ligand_residue_id='UNK', cutoff_distance=5)
    t1 = time.time()
    print(f"[1/4] 口袋提取完成: {t1-t0:.2f}s")
    print(f"       输出: {pocket_path_npy}")
except Exception as e:
    print(f"[1/4] 口袋提取失败: {e}")
    sys.exit(1)

# 步骤2: 读取SMILES/构象生成/对齐
print("[2/4] 开始构象生成和对齐...")
try:
    min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(dirname, num_confs=num_confs, ncpu=ncpu)
    t2 = time.time()
    print(f"[2/4] 构象生成+对齐完成: {t2-t1:.2f}s")
    print(f"       生成分子数: {len(df)}")
    print(f"       边界: x[{min_x},{max_x}] y[{min_y},{max_y}] z[{min_z},{max_z}]")
except Exception as e:
    print(f"[2/4] 构象生成失败: {e}")
    sys.exit(1)

# 步骤3: 顶点生成
print("[3/4] 开始顶点生成...")
try:
    vertices = vertices_gen(pocket_path_npy, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=ncpu)
    t3 = time.time()
    print(f"[3/4] 顶点生成完成: {t3-t2:.2f}s")
    print(f"       顶点数: {len(vertices)}")
except Exception as e:
    print(f"[3/4] 顶点生成失败: {e}")
    sys.exit(1)

# 步骤4: 特征化+tensor生成
print("[4/4] 开始特征化和tensor生成...")
try:
    ind_att_pd_path, tensor_path, state_path = featurizer_new(dirname, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=ncpu)
    t4 = time.time()
    print(f"[4/4] 特征化+tensor完成: {t4-t3:.2f}s")
except Exception as e:
    print(f"[4/4] 特征化失败: {e}")
    sys.exit(1)

print()
print(f"=== 总耗时: {t4-t0:.2f}s ===")
print(f"输出文件:")
print(f"  - ind_att_pd: {ind_att_pd_path}")
print(f"  - tensor.pkl: {tensor_path}")
print(f"  - state.csv: {state_path}")

# 检查输出文件
print()
print("=== 输出文件验证 ===")
for f in [ind_att_pd_path, tensor_path, state_path]:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  ✓ {f} ({size} bytes)")
    else:
        print(f"  ✗ {f} (未找到)")
