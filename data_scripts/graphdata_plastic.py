import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# read the raw data
data_loc = '/work/hdd/bdsy/wzhong/LUG_3D/Data/'
xyz = np.load(os.path.join(data_loc, 'xyz_coords.npy'))    # (29852, 3)
loading = np.load(os.path.join(data_loc, 'data_amp_10K.npy'))    # (10000, 101)
# strain = np.load(os.path.join(data_loc, 'peeq_all_new_10K.npy'))    # (10000, 29852)
stress = np.load(os.path.join(data_loc, 'ystress_all_new_10K.npy')) * 1e-6    # (10000, 29852)

'''
Data Preprocessing
'''
# Compute the 95th percentile threshold
# strain_threshold = np.percentile(strain.flatten(), 95)  
stress_threshold = np.percentile(stress.flatten(), 95)  
# cutoff
stress[stress>stress_threshold] = stress_threshold
# strain[strain>strain_threshold] = strain_threshold

def graph_rep_generate(points, stress, save_path, debug):
    '''
    Convert data into a graph representation and save it.
    points (M, 3): 3D coordinates of points
    strain (B, M): Strain values for each sample
    stress (B, M): Stress values for each sample
    save_path (str): Path to save the processed data
    '''
    
    # Normalize data
    points_min, points_max = points.min(axis=0), points.max(axis=0)
    stress_min, stress_max = stress.min(), stress.max()
    print(stress_min, stress_max)
    assert 1==2
    
    points_norm =  points * 10 # (points - points_min) / (points_max - points_min)
    stress_norm = (stress - stress_min) / (stress_max - stress_min)
    
    # Construct edge connections using k-nearest neighbors (k=5)
    k = 5
    tree = KDTree(points)
    edge_index = []
    edge_attr = []
    print('Processing the graph')
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
    for i in range(len(stress_norm)):
        if i % 10 == 0:
            print('Curent sample graph ID:', i)
        x = torch.tensor(points_norm, dtype=torch.float)
        y = torch.tensor(stress_norm[i,:], dtype=torch.float).unsqueeze(-1)
        par = torch.tensor(loading[i], dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
            param = par,
            added_edge_index=added_edge_index, added_edge_attr=added_edge_attr)
        torch.save(graph_data, save_path + 'graph{}.pt'.format(i))
        if i > 200:
            if debug:
                assert 1==2

# Specify the output file
save_file = '/work/hdd/bdsy/wzhong/processed_data/plastic/'
graph_rep_generate(xyz, stress, save_file, debug=False)