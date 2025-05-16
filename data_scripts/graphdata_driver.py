import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pickle as pkl
import pyvista as pv
import time
from sklearn.cluster import KMeans
import argparse
import os
import re

parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--id', type=int, default=1)
args = parser.parse_args()
folder_id = args.id

# read the raw data
data_loc = '/work/hdd/bdsy/wzhong/driver/PressureVTK/'
vtk_files = []
for i in range(folder_id,folder_id+1):
    subdir = 'D{}'.format(i)
    subdir_path = os.path.join(data_loc, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith('.vtk'):
                vtk_files.append(os.path.join(subdir_path, filename))
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
param_dict = read_csv_to_experiment_dict(r'./data_scripts/DrivAerNet_ParametricData.csv')

# extract data from vtk files
vertices = []
nodal_stress = []
points_bc = []
Links = []
Gids = []
params_list = []
for i in range(len(vtk_files)):
    try:
        mesh = pv.read(vtk_files[i])
        points = mesh.points
        pressure = mesh.point_data["p"]
    except:
        continue
    
    if i % 10 == 0:
        print(i, np.amax(pressure), np.amin(pressure))

    # extract the params
    experiment_id = os.path.splitext(os.path.basename(vtk_files[i]))[0]
    if experiment_id in param_dict.keys():
        params = param_dict[experiment_id]["param_vector"]
        Gid = param_dict[experiment_id]["index"]
    else:
        continue

    if i % 10 == 0:
        print(i)
    
    # store graph id
    Gids.append(Gid)
    params_list.append(params)

    '''
    farthest point processing is very slow
    '''
    '''
    We use farthest point
    '''
    # num_points = points.shape[0]
    # control_pc = [points[np.random.randint(num_points)]]
    # num_control = int(0.01 * num_points)
    # # Track distances from control points to all others
    # distances = np.full(num_points, np.inf)
    # for _ in range(num_control - 1):
    #     # Compute distance from the latest control point to all others
    #     last_cp = np.mean(np.array(control_pc), 0)
    #     dists = np.linalg.norm(points - last_cp, axis=1)
    #     # Keep the minimum distance to any control point so far
    #     distances = np.minimum(distances, dists)
    #     # Select the point farthest from current control set
    #     next_idx = np.argmax(distances)
    #     control_pc.append(points[next_idx])
    # # Convert list to array
    # control_pc = np.array(control_pc)
    control_pc = points.copy()[np.random.randint(0, points.shape[0], int(0.01 * points.shape[0])),:]

    # # extract cell
    # cells = mesh.faces
    # cell_list = []
    # j = 0
    # while j < len(cells):
    #     nn = cells[j]
    #     cell_list.append(cells[j+1:j+nn+1])
    #     j = j + nn + 1

    # store the data
    vertices.append(points)
    nodal_stress.append(pressure)
    points_bc.append(control_pc)
    # Links.append(cell_list)
print('Percentage of stored data:', len(params_list)/len(vtk_files))
assert 1==2


# data normalization
def min_max_scale(vertices, nodal_stress, points_bc, params_list):

    # normalize params_list
    params_list = np.array(params_list)
    max_params_list = np.amax(params_list, 0, keepdims=True)
    min_params_list = np.amin(params_list, 0, keepdims=True)
    params_list = (params_list - min_params_list) / (max_params_list - min_params_list)

    # compute the min and max
    min_stress_features = -20000  # -10142, -27292
    max_stress_features = 10000  # 700, 6016

    # Min-Max normalization
    for i in range(len(vertices)):
        nodal_stress[i] = (nodal_stress[i] - min_stress_features) / (max_stress_features - min_stress_features)
    
    return vertices, nodal_stress, points_bc, params_list

vertices, nodal_stress, points_bc, params_list = min_max_scale(vertices, nodal_stress, points_bc, params_list)

# graph process
def graph_rep_generate(points_list, nodal_stress, points_bc, params_list, Gids, save_path, stored_graph_ids, debug=True):
    '''
    Convert data into a graph representation and save it.
    points (M, 3): 3D coordinates of points
    strain (B, M): Strain values for each sample
    stress (B, M): Stress values for each sample
    save_path (str): Path to save the processed data
    '''
    num_graph = len(points_list)
    for iid in range(num_graph):

        # skip if we processed it:
        if Gids[iid] in stored_graph_ids:
            print('this graph is processed.')
            continue

        if iid % 1 == 0:
            print('Curent sample graph ID:', iid)

        # extract points
        points = points_list[iid]
        st = time.time()
        tree = KDTree(points)
        edge_index = []
        edge_attr = []
        k=3
        for i in range(points.shape[0]):
            distances, indices = tree.query(points[i], k=k+1)  # k+1 to include itself
            for j in range(1, k+1):  # Exclude self-connection
                neighbor = indices[j]
                edge_index.append([i, neighbor])
                edge_index.append([neighbor, i])
                edge_attr.append(np.concatenate((points[i], points[neighbor]), 0))
                edge_attr.append(np.concatenate((points[neighbor], points[i]), 0))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()    # (2,E)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)    # (E, 6)
        et = time.time()

        # randomly create long-range link
        num_added_edges = int(points.shape[0])
        random_edge_node_index = np.random.randint(0, high=int(points.shape[0]), size=(2,num_added_edges))
        added_edge_index = []
        added_edge_attr = []
        for i in range(num_added_edges):
            node1 = random_edge_node_index[0,i]
            node2 = random_edge_node_index[1,i]
            added_edge_index.append([node1, node2])
            added_edge_index.append([node2, node1])
            added_edge_attr.append(np.concatenate((points[node1], points[node2]), 0))
            added_edge_attr.append(np.concatenate((points[node2], points[node1]), 0))
        added_edge_index = torch.tensor(added_edge_index, dtype=torch.long).t().contiguous()    # (2,E)
        added_edge_attr = torch.tensor(added_edge_attr, dtype=torch.float)    # (E, 6)

        # Create PyTorch Geometric graph representation
        x = torch.tensor(points, dtype=torch.float)
        y = torch.tensor(nodal_stress[iid], dtype=torch.float).unsqueeze(-1)
        control_points = torch.tensor(points_bc[iid], dtype=torch.float)
        param = torch.tensor(params_list[iid,:], dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
            param = param,
            control_points = control_points,
            added_edge_index=added_edge_index, added_edge_attr=added_edge_attr)
        torch.save(graph_data, save_path + 'graph{}.pt'.format(Gids[iid]))
        if iid > 20:
            if debug:
                assert 1==2

# Specify the output file
save_file = '/work/hdd/bdsy/wzhong/processed_data/driver/'

# check processed graph
def check_processed_graph_ids(folder_path, file_prefix="graph", file_extension=".pt"):
    graph_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith(file_prefix) and f.endswith(file_extension)
    ]
    graph_paths.sort()  # Optional: ensure consistent order
    # Extract numeric IDs from filenames
    graph_ids = []
    for path in graph_paths:
        filename = os.path.basename(path)
        match = re.search(rf"{file_prefix}(\d+){file_extension}", filename)
        if match:
            graph_ids.append(int(match.group(1)))

    return set(graph_ids)

stored_graph_ids = check_processed_graph_ids(save_file)
graph_rep_generate(vertices, nodal_stress, points_bc, params_list, Gids, save_file, stored_graph_ids, debug=False)