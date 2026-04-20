import os
import sys
import glob
import shutil
import pickle
import pandas as pd

PROJECT_ROOT = r"E:\pharm3d"
JOB_DIR = r"E:\pharm3d\one_molecule_run\job_0001"
FIXED_INDEX_CSV = r"E:\pharm3d\00011734250639\ind_att_pd.csv"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pharm"))

def import_pipeline():
    # 兼容不同导入路径
    try:
        from embed.pocket import get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen
        from embed.utils import getSimplifiedMatrix
        return get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen, getSimplifiedMatrix
    except Exception:
        try:
            from pharm.embed.pocket import get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen
            from pharm.embed.utils import getSimplifiedMatrix
            return get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen, getSimplifiedMatrix
        except Exception:
            from embed import get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen
            from embed.utils import getSimplifiedMatrix
            return get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen, getSimplifiedMatrix

def main():
    print("=== 单分子固定索引兼容版前处理开始 ===")
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("JOB_DIR =", JOB_DIR)
    print("FIXED_INDEX_CSV =", FIXED_INDEX_CSV)

    required_files = [
        os.path.join(JOB_DIR, "template_known_mols.smi"),
        os.path.join(JOB_DIR, "box_info.txt"),
        os.path.join(JOB_DIR, "crystal_ligand.mol2"),
        os.path.join(JOB_DIR, "template_complex.pdb"),
        FIXED_INDEX_CSV,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少文件: {path}")

    get_protein_ligand_neighbors, featurizer_new, read_smiles, vertices_gen, getSimplifiedMatrix = import_pipeline()

    ncpu = 1
    num_confs = 5

    print("Step 1: 提取蛋白-配体邻近区域")
    pocket_path_npy = get_protein_ligand_neighbors(JOB_DIR, ligand_residue_id='UNK', cutoff_distance=5)
    print("pocket_path_npy =", pocket_path_npy)

    print("Step 2: 读取单分子 SMILES 并生成构象")
    min_x, max_x, min_y, max_y, min_z, max_z, df = read_smiles(JOB_DIR, num_confs=num_confs, ncpu=ncpu)
    print("box =", (min_x, max_x, min_y, max_y, min_z, max_z))
    print("df shape =", getattr(df, "shape", None))

    print("Step 3: 生成 pocket vertices")
    vertices = vertices_gen(pocket_path_npy, min_x, max_x, min_y, max_y, min_z, max_z, ncpu=ncpu)
    print("vertices generated")

    print("Step 4: 跑标准 featurizer_new，生成中间 featMatrix 与动态 tensor")
    generated_index_path, generated_tensor_path, state_path = featurizer_new(
        JOB_DIR, min_x, max_x, min_y, max_y, min_z, max_z, df, vertices, ncpu=ncpu
    )

    print("generated dynamic ind_att_pd =", generated_index_path)
    print("generated dynamic tensor.pkl =", generated_tensor_path)
    print("state.csv =", state_path)

    # 备份动态生成结果
    dynamic_index_backup = os.path.join(JOB_DIR, "ind_att_pd_generated_dynamic.csv")
    dynamic_tensor_backup = os.path.join(JOB_DIR, "tensor_generated_dynamic.pkl")

    if os.path.exists(generated_index_path):
        shutil.copy2(generated_index_path, dynamic_index_backup)
        print("已备份动态 index ->", dynamic_index_backup)

    if os.path.exists(generated_tensor_path):
        shutil.copy2(generated_tensor_path, dynamic_tensor_backup)
        print("已备份动态 tensor ->", dynamic_tensor_backup)

    print("Step 5: 读取训练时固定 ind_att_pd.csv")
    fixed_ind_att = pd.read_csv(FIXED_INDEX_CSV)
    print("fixed_ind_att shape =", fixed_ind_att.shape)

    feat_files = sorted(glob.glob(os.path.join(JOB_DIR, "featMatrix_*.pkl")))
    if not feat_files:
        raise FileNotFoundError("没有找到 featMatrix_*.pkl，中间特征矩阵未生成。")

    print("找到中间特征文件:")
    for fp in feat_files:
        print(" -", fp)

    print("Step 6: 用训练时固定 index 重新展平，得到与现有 MLP 兼容的 tensor")
    fixed_tensor = []

    for fp in feat_files:
        with open(fp, "rb") as f:
            feat_list = pickle.load(f)

        print(f"读取 {fp}，其中包含 {len(feat_list)} 个条目")

        for one_feat_matrix in feat_list:
            vec = getSimplifiedMatrix(one_feat_matrix, ind_att_pd=fixed_ind_att)
            fixed_tensor.append(vec)

    fixed_index_copy = os.path.join(JOB_DIR, "ind_att_pd_fixed_from_training.csv")
    fixed_tensor_copy = os.path.join(JOB_DIR, "tensor_fixed_from_training.pkl")

    fixed_ind_att.to_csv(fixed_index_copy, index=False)
    with open(fixed_tensor_copy, "wb") as f:
        pickle.dump(fixed_tensor, f)

    # 覆盖 canonical 文件名，方便后续直接按 tensor.pkl 使用
    shutil.copy2(fixed_index_copy, os.path.join(JOB_DIR, "ind_att_pd.csv"))
    with open(os.path.join(JOB_DIR, "tensor.pkl"), "wb") as f:
        pickle.dump(fixed_tensor, f)

    print("Step 7: 输出摘要")
    print("fixed index saved to =", fixed_index_copy)
    print("fixed tensor saved to =", fixed_tensor_copy)
    print("canonical tensor.pkl overwritten =", os.path.join(JOB_DIR, "tensor.pkl"))

    print("tensor rows =", len(fixed_tensor))
    if len(fixed_tensor) > 0:
        try:
            print("tensor width =", len(fixed_tensor[0]))
        except Exception as e:
            print("无法获取 tensor width:", e)

    print("=== 完成：已生成与现有训练 index 对齐的 tensor.pkl ===")

if __name__ == "__main__":
    main()
