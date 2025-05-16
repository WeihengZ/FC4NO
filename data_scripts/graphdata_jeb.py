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
save_file = os.path.join(data_root, "GraphData_Standardized/")
os.makedirs(save_file, exist_ok=True)

# === Configuration ===
sample_ids = sorted([f.split('.')[0] for f in os.listdir(field_mesh_path) if f.endswith('.h5')])
nodal_variable_name = "ver_stress(MPa)"  # Modify as needed

# === Load and store mesh samples ===
vertices, nodal_stress = [], []
all_used_indices = []

for sid in tqdm(sample_ids):
    field_file = os.path.join(field_mesh_path, f"{sid}.h5")
    volume_file = os.path.join(volume_mesh_path, f"{sid}.vtk")
    if not os.path.exists(field_file) or not os.path.exists(volume_file):
        continue

    with h5py.File(field_file, 'r') as f:
        nodal_var = f['nodal_variables'][nodal_variable_name][:]
    nodal_var = np.append(nodal_var, [0]*5)  # pad to match VTK

    vtk_mesh = pv.read(volume_file)
    original_points = vtk_mesh.points
    original_cells = vtk_mesh.cells.reshape(-1, 11)

    reduced_cells = np.array([[4] + list(cell[1:5]) for cell in original_cells])
    used_points = np.unique(reduced_cells[:, 1:])
    point_mapping = {old: new for new, old in enumerate(used_points)}

    new_points = original_points[used_points]
    new_nodal_var = nodal_var[used_points]

    vertices.append(new_points)
    nodal_stress.append(new_nodal_var)
    all_used_indices.append((reduced_cells, point_mapping, used_points, sid))

# === Standardization ===
def standardize(vertices, nodal_stress):
    all_stresses = np.hstack(nodal_stress)
    mean_stress = np.mean(all_stresses)
    std_stress = np.std(all_stresses)

    for i in range(len(vertices)):
        nodal_stress[i] = (nodal_stress[i] - mean_stress) / std_stress

    return vertices, nodal_stress

vertices, nodal_stress = standardize(vertices, nodal_stress)

# === Construct graphs using mesh topology ===
for i, (cells, point_mapping, used_points, sid) in enumerate(tqdm(all_used_indices)):
    points = vertices[i]
    stress = nodal_stress[i]

    # Convert cell definitions to edge list
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
        torch.cat([torch.tensor(points[edge_index[0]]), torch.tensor(points[edge_index[1]])], dim=-1)
    ], dim=-1)

    x = torch.tensor(points, dtype=torch.float)
    y = torch.tensor(stress, dtype=torch.float).unsqueeze(-1)
    control_points = x[np.random.choice(x.shape[0], 128, replace=False)]  # dummy BC

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                control_points=control_points)
    torch.save(data, os.path.join(save_file, f"graph_{sid}.pt"))
