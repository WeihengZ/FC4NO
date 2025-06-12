import os
import numpy as np
import torch
from torch_geometric.data import Data
import pyvista as pv
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict

# === Paths ===
data_loc = "/taiga/illinois/eng/cee/meidani/Vincent/driver_plus/PressureVTK/"
save_file = "/taiga/illinois/eng/cee/meidani/Vincent/driver_plus/Processed/"
os.makedirs(save_file, exist_ok=True)

# === Gather all .vtk files recursively ===
vtk_files = []
for root, _, files in os.walk(data_loc):
    for file in files:
        if file.endswith('.vtk'):
            vtk_files.append(os.path.join(root, file))
print("Total .vtk files found:", len(vtk_files))

# read the parametric representation
import csv
def read_csv_to_experiment_dict(csv_file_path):
    experiment_data = {}
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[1:]  # skip "Experiment" column in headers
        for idx, row in enumerate(reader):
            experiment_id = row[0]
            values = list(map(float, row[1:]))  # convert all feature values to float
            experiment_data[experiment_id] = {
                "param_vector": values,
                "index": idx
            }
    return experiment_data
param_dict = read_csv_to_experiment_dict(r'./DrivAerNet_ParametricData.csv')
scale_factor = np.load(save_file + "mean_std.npz")
mean_stress, std_stress = scale_factor['mean'], scale_factor['std']

for file in tqdm(vtk_files):
    mesh = pv.read(file)
    points = mesh.points
    pressure = mesh.point_data["p"]
    sid = os.path.splitext(os.path.basename(file))[0]

    # Normalize pressure
    norm_pressure = (pressure - mean_stress) / std_stress

    param = param_dict[sid]["param_vector"]
    data = Data(x=points, y=norm_pressure, param=param)
    torch.save(data, os.path.join(save_file, f"graph_{sid}.pt"))