import os
import pickle

tensor_path = r"E:\pharm3d\one_molecule_run\job_0001\tensor.pkl"

print("tensor exists:", os.path.exists(tensor_path))

with open(tensor_path, "rb") as f:
    obj = pickle.load(f)

print("python type:", type(obj))
print("outer len:", len(obj))

if len(obj) > 0:
    first = obj[0]
    print("first element type:", type(first))
    try:
        print("first element len:", len(first))
    except Exception as e:
        print("cannot get first element len:", e)
