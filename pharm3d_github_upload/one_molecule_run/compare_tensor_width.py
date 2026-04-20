import pandas as pd
import pickle

index_path = r"E:\pharm3d\00011734250639\ind_att_pd.csv"
tensor_path = r"E:\pharm3d\one_molecule_run\job_0001\tensor.pkl"

idx = pd.read_csv(index_path)

with open(tensor_path, "rb") as f:
    tensor = pickle.load(f)

training_index_rows = len(idx)
tensor_width = len(tensor[0]) if len(tensor) > 0 else 0

print("training index rows =", training_index_rows)
print("tensor width =", tensor_width)
print("compatible =", training_index_rows == tensor_width)
