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

parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--id', type=int, default=1)
args = parser.parse_args()
folder_id = args.id

# read the raw data
data_loc = '/work/hdd/bdsy/wzhong/driver_plus/PressureVTK2/'
vtk_files = []
for i in range(folder_id,folder_id+1):
    subdir = 'F_D_WM_WW_{}'.format(i+1)
    subdir_path = os.path.join(data_loc, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith('.vtk'):
                vtk_files.append(os.path.join(subdir_path, filename))
print("Total .vtk files found:", len(vtk_files))

# extract data from vtk files
vertices = []
nodal_stress = []
points_bc = []
Links = []
Gids = []
for i in range(len(vtk_files)):
    mesh = pv.read(vtk_files[i])
    points = mesh.points
    pressure = mesh.point_data["p"]

    if i % 10 == 0:
        print(i, np.amax(pressure), np.amin(pressure))
    
    # extract graph id
    Gid = int(vtk_files[i][-8:-4])
    Gids.append(Gid)

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

assert 1==2

# data normalization
def min_max_scale(vertices, nodal_stress, points_bc):

    # compute the min and max
    min_stress_features = -20000  # -10142, -27292
    max_stress_features = 10000  # 700, 6016

    # Min-Max normalization
    for i in range(len(vertices)):
        nodal_stress[i] = (nodal_stress[i] - min_stress_features) / (max_stress_features - min_stress_features)
    
    return vertices, nodal_stress, points_bc

vertices, nodal_stress, points_bc = min_max_scale(vertices, nodal_stress, points_bc)

# graph process
def graph_rep_generate(points_list, nodal_stress, points_bc, Gids, save_path, debug=True):
    '''
    Convert data into a graph representation and save it.
    points (M, 3): 3D coordinates of points
    strain (B, M): Strain values for each sample
    stress (B, M): Stress values for each sample
    save_path (str): Path to save the processed data
    '''
    num_graph = len(points_list)
    for iid in range(num_graph):

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
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
            control_points = control_points,
            added_edge_index=added_edge_index, added_edge_attr=added_edge_attr)
        torch.save(graph_data, save_path + 'graph{}.pt'.format(Gids[iid]))
        if iid > 20:
            if debug:
                assert 1==2

# Specify the output file
save_file = '/work/hdd/bdsy/wzhong/processed_data/driver_plus/'
graph_rep_generate(vertices, nodal_stress, points_bc, Gids, save_file, debug=False)