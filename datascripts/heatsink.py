import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm

# raw data file path
data_loc = '/taiga/illinois/eng/cee/meidani/Vincent/heatsink/'
# saved data path
save_path = '/taiga/illinois/eng/cee/meidani/Vincent/heatsink/Processed/'

# data normalization with min-max normalization
def min_max_scale(params, vertices, temperatures, points_bc):

    # compute the min and max
    all_vertices = np.vstack(vertices)
    all_stress = np.hstack(temperatures)
    min_xyz_features = np.amin(all_vertices, 0, keepdims=True)
    max_xyz_features = np.amax(all_vertices, 0, keepdims=True)
    min_stress_features = np.amin(all_stress)
    max_stress_features = np.amax(all_stress)
    min_param_features = np.amin(params, 0, keepdims=True)
    max_param_features = np.amax(params, 0, keepdims=True)

    # Min-Max normalization
    params = (params - min_param_features) / (max_param_features - min_param_features)
    for i in range(len(vertices)):
        vertices[i] = (vertices[i] - min_xyz_features) / (max_xyz_features - min_xyz_features)
        points_bc[i] = (points_bc[i] - min_xyz_features) / (max_xyz_features - min_xyz_features)
        temperatures[i] = (temperatures[i] - min_stress_features) / (max_stress_features - min_stress_features)
    
    return params, vertices, temperatures, points_bc

# graph process
def graph_rep_generate(params, points_list, nodal_stress, points_bc, save_path):
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
            geo_param = par,
            control_points = control_points,
            added_edge_index=added_edge_index, added_edge_attr=added_edge_attr)
        torch.save(graph_data, save_path + 'graph{}.pt'.format(iid))

# read the raw data
data = sio.loadmat(os.path.join(data_loc, 'heat_sink_fem_simulation_results.mat'))['all_results'][0][0]
params = []
vertices = []
temperatures = []
points_bc = []
for i in tqdm(range(len(data))):
    datap = data[i][0][0]

    # extract parameter
    W = datap[0][0][0]    # fin width
    H = datap[1][0][0]    # fin height
    t = datap[2][0][0]    # fin thickness
    nf = datap[3][0][0]   # number of fins
    params.append([W, H, t, nf])

    # extract point cloud
    pc = datap[4].T    # (M, 3)

    # extract temperature distribution
    temp = datap[5].squeeze(-1)    # (M)

    # extract the fin top as the control points
    fin_top = np.amax(pc[:,-1])
    fin_top_node_id = np.argwhere(pc[:,-1]==fin_top)[:,0]
    fin_top_coor = pc[fin_top_node_id,:]

    # store the data
    vertices.append(pc)
    temperatures.append(temp)
    points_bc.append(fin_top_coor)
params = np.array(params)

# normalize the data
params, vertices, temperatures, points_bc = min_max_scale(params, vertices, temperatures, points_bc)

# save graphs
graph_rep_generate(params, vertices, temperatures, points_bc, save_path)