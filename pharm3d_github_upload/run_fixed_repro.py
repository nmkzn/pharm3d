import os
import re
from pharm.embed.train import pharm_train

JOB_DIR = os.path.expanduser("~/00011734250639")
BOX_PATH = os.path.join(JOB_DIR, "box_info.txt")
TENSOR_PATH = os.path.join(JOB_DIR, "tensor.pkl")
STATE_PATH = os.path.join(JOB_DIR, "state.csv")

required_files = [
    BOX_PATH,
    TENSOR_PATH,
    STATE_PATH,
    os.path.join(JOB_DIR, "train_idx.npy"),
    os.path.join(JOB_DIR, "val_idx.npy"),
    os.path.join(JOB_DIR, "test_idx.npy"),
    os.path.join(JOB_DIR, "ind_att_pd.csv"),
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件: {path}")

with open(BOX_PATH, "r", encoding="utf-8") as f:
    text = f.read()

nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', text)]
if len(nums) < 6:
    raise ValueError(f"box_info.txt 中没有解析出 6 个边界值，实际得到: {nums}")

min_x, max_x, min_y, max_y, min_z, max_z = nums[:6]

print("JOB_DIR =", JOB_DIR)
print("box bounds =", min_x, max_x, min_y, max_y, min_z, max_z)

out = pharm_train(
    JOB_DIR,
    min_x, max_x,
    min_y, max_y,
    min_z, max_z,
    TENSOR_PATH,
    STATE_PATH,
)

print("训练完成，返回文件 =", out)
