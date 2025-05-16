import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.io as sio

# read the raw data
data_loc = '/work/hdd/bdsy/wzhong/heatsink/'
data = sio.loadmat(os.path.join(data_loc, 'heat_sink_fem_simulation_results.mat'))['all_results'][0][0]
params = []
vertices = []
nodal_stress = []
points_bc = []
for i in range(len(data)):
    datap = data[i][0][0]

    # extract parameter
    W = datap[0][0][0]
    H = datap[1][0][0]
    t = datap[2][0][0]
    nf = datap[3][0][0]
    params.append([W, H, t, nf])

    # extract point cloud
    pc = datap[4].T    # (M, 3)

    # extract temperature
    temp = datap[5].squeeze(-1)    # (M)

    # extract the fin top as the control points
    fin_top = np.amax(pc[:,-1])
    fin_top_node_id = np.argwhere(pc[:,-1]==fin_top)[:,0]
    fin_top_coor = pc[fin_top_node_id,:]

    # store the data
    vertices.append(pc)
    nodal_stress.append(temp)
    points_bc.append(fin_top_coor)
params = np.array(params)

# data normalization
def min_max_scale(params, vertices, nodal_stress, points_bc):

    # compute the min and max
    all_vertices = np.vstack(vertices)
    all_stress = np.hstack(nodal_stress)
    min_xyz_features = np.amin(all_vertices, 0, keepdims=True)
    max_xyz_features = np.amax(all_vertices, 0, keepdims=True)
    min_stress_features = np.amin(all_stress)
    max_stress_features = np.amax(all_stress)
    min_param_features = np.amin(params, 0, keepdims=True)
    max_param_features = np.amax(params, 0, keepdims=True)

    print(min_stress_features, max_stress_features)
    assert 1==2

    # Min-Max normalization
    params = (params - min_param_features) / (max_param_features - min_param_features)
    for i in range(len(vertices)):
        vertices[i] = (vertices[i] - min_xyz_features) / (max_xyz_features - min_xyz_features)
        points_bc[i] = (points_bc[i] - min_xyz_features) / (max_xyz_features - min_xyz_features)
        nodal_stress[i] = (nodal_stress[i] - min_stress_features) / (max_stress_features - min_stress_features)
    
    return params, vertices, nodal_stress, points_bc

params, vertices, nodal_stress, points_bc = min_max_scale(params, vertices, nodal_stress, points_bc)


# graph process
def graph_rep_generate(params, points_list, nodal_stress, points_bc, save_path, debug=True):
    '''
    Convert data into a graph representation and save it.
    points (M, 3): 3D coordinates of points
    strain (B, M): Strain values for each sample
    stress (B, M): Stress values for each sample
    save_path (str): Path to save the processed data
    '''
    num_graph = len(points_list)
    for iid in range(num_graph):

        if iid % 10 == 0:
            print('Curent sample graph ID:', iid)

        # extract points
        points = points_list[iid]
    
        # Construct edge connections using k-nearest neighbors (k=5)
        k = 5
        tree = KDTree(points)
        edge_index = []
        edge_attr = []
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

        # randomly create long-range link
        num_edges = edge_index.shape[1]
        num_added_edges = int(0.01 * 0.5 * num_edges)
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
        par = torch.tensor(params[iid], dtype=torch.float)
        control_points = torch.tensor(points_bc[iid], dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
            param = par,
            control_points = control_points,
            added_edge_index=added_edge_index, added_edge_attr=added_edge_attr)
        torch.save(graph_data, save_path + 'graph{}.pt'.format(iid))
        if iid > 50:
            if debug:
                assert 1==2

# Specify the output file
save_file = '/work/hdd/bdsy/wzhong/processed_data/heatsink/'
graph_rep_generate(params, vertices, nodal_stress, points_bc, save_file, debug=False)