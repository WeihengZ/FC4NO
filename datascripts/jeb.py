import os
import numpy as np
import h5py
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# === Set data paths ===
data_root = "/taiga/illinois/eng/cee/meidani/Vincent/JEB"
field_mesh_path = os.path.join(data_root, "FieldMesh")
volume_mesh_path = os.path.join(data_root, "VolumeMesh")
save_file = os.path.join(data_root, "Processed/")
os.makedirs(save_file, exist_ok=True)

# === Configuration ===
sample_ids = sorted([f.split('.')[0] for f in os.listdir(field_mesh_path) if f.endswith('.h5')])
nodal_variable_name = "ver_stress(MPa)"  # Modify as needed

# === Compute and save mean/std if not present ===
mean_std_file = os.path.join(save_file, "mean_std.npz")
if not os.path.exists(mean_std_file):
    all_stress = []
    for sid in tqdm(sample_ids, desc='Collecting stress for mean/std'):
        field_file = os.path.join(field_mesh_path, f"{sid}.h5")
        if not os.path.exists(field_file):
            continue
        with h5py.File(field_file, 'r') as f:
            nodal_var = f['nodal_variables'][nodal_variable_name][:]
        all_stress.append(nodal_var)
    all_stress = np.concatenate(all_stress)
    mean_stress = all_stress.mean()
    std_stress = all_stress.std()
    np.savez(mean_std_file, mean=mean_stress, std=std_stress)

# read the mean and std
mean_stress = np.load(mean_std_file)['mean']
std_stress = np.load(mean_std_file)['std']

# Initialize running min and max for x, y, z
running_min = None
running_max = None

# === Load and store mesh samples ===
for sid in tqdm(sample_ids):
    field_file = os.path.join(field_mesh_path, f"{sid}.h5")
    volume_file = os.path.join(volume_mesh_path, f"{sid}.vtk")

    if not os.path.exists(field_file) or not os.path.exists(volume_file):
        continue

    with h5py.File(field_file, 'r') as f:
        nodal_var = f['nodal_variables'][nodal_variable_name][:]
    nodal_var = np.append(nodal_var, [0]*5)  # pad to match VTK

    vtk_mesh = pv.read(volume_file, file_format="vtk")
    original_points = vtk_mesh.points
    original_cells = vtk_mesh.cells.reshape(-1, 11)

    # link information
    cells = np.array([[4] + list(cell[1:5]) for cell in original_cells])
    used_points = np.unique(cells[:, 1:])
    point_mapping = {old: new for new, old in enumerate(used_points)}

    # nodal points
    points = original_points[used_points] / 20
    nodal_stress = nodal_var[used_points]

    # standardize
    nodal_stress = (nodal_stress - mean_stress) / std_stress

    # create the graph
    edge_set = set()
    for cell in cells:
        verts = [point_mapping[idx] for idx in cell[1:5]]
        for a in range(4):
            for b in range(a+1, 4):
                edge_set.add((verts[a], verts[b]))
                edge_set.add((verts[b], verts[a]))  # undirected
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()  # shape (2, E)

    # edge_attr: concat coordinates of endpoints
    edge_attr = torch.cat([
        torch.cat([torch.tensor(points[edge_index[0]], dtype=torch.float), 
            torch.tensor(points[edge_index[1]], dtype=torch.float)], dim=-1)
    ], dim=-1)
    
    points = torch.tensor(points, dtype=torch.float)
    nodal_stress = torch.tensor(nodal_stress, dtype=torch.float)

    # save the data
    data = Data(x=points, edge_index=edge_index, edge_attr=edge_attr, y=nodal_stress)
    torch.save(data, os.path.join(save_file, f"graph_{sid}.pt"))

    # Update running min and max
    pts = torch.tensor(points)
    if running_min is None:
        running_min = pts.min(dim=0).values
        running_max = pts.max(dim=0).values
    else:
        running_min = torch.minimum(running_min, pts.min(dim=0).values)
        running_max = torch.maximum(running_max, pts.max(dim=0).values)
    x_min, y_min, z_min = running_min.tolist()
    x_max, y_max, z_max = running_max.tolist()
    print(f"After {sid}: x: min={x_min}, max={x_max}; y: min={y_min}, max={y_max}; z: min={z_min}, max={z_max}")
