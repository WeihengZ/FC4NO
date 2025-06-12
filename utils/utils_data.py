from torch_geometric.data import Dataset, Data, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TensorDataLoader
import time
import matplotlib.pyplot as plt
import os
import pickle as pkl
import time

def get_graph_paths(folder_path, file_prefix="graph", file_extension=".pt"):
    graph_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith(file_prefix) and f.endswith(file_extension)
    ]
    graph_paths.sort()  # Optional: ensure consistent order
    return graph_paths

class GraphDataset(Dataset):
    def __init__(self, graph_paths, params_data=None):
        super(GraphDataset, self).__init__()

        # Graph file paths
        self.graph_paths = graph_paths
        self.return_param = False
        if params_data is not None:
            self.return_param = True
            self.params_data = params_data
            assert len(self.graph_paths) == len(self.params_data), "Graph paths and params data must have the same length."

    def __len__(self):
        """Returns the number of graphs available in the dataset."""
        return len(self.graph_paths)
    
    def len(self): 
        return len(self.graph_paths)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        """Loads a graph from file and optionally returns its grid shape."""
        # single_graph_path = self.graph_paths[idx]
        # graph = torch.load(single_graph_path, weights_only=False)

        single_graph_path = self.graph_paths[idx]
        try:
            graph = torch.load(single_graph_path, weights_only=False)
        except Exception as e:
            print(f"Failed to load {single_graph_path}: {e}")
            # Option 1: skip or return dummy data
            return self.__getitem__((idx + 1) % len(self.graph_paths))
        
        # check the name of the parameters
        if 'geo_param' in graph.keys():
            graph.geo_params = graph.geo_param
        if 'load_param' in graph.keys():
            graph.load_params = graph.load_param

        # add the params if needed
        if self.return_param:
            graph.param = torch.from_numpy(self.params_data[idx]).float()

        return graph, single_graph_path

def create_data_loaders(graph_folder_path, batch_size=1):
    """
    Creates train, validation, and test data loaders.

    Parameters:
    - graph_paths (list): List of file paths to graph data.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - train_loader, val_loader, test_loader: DataLoader objects for each split.
    """
    graph_paths = get_graph_paths(graph_folder_path)
    num_samples = len(graph_paths)

    # Create dataset
    dataset = GraphDataset(graph_paths)

    # Define train, validation, and test split indices
    train_size = 10 # int(0.6 * num_samples)
    val_size = 2 # int(0.1 * num_samples)
    test_size = 1 # int(num_samples) - train_size - val_size

    # Split dataset
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=True,
        num_workers=0,   
        pin_memory=False)
    val_loader = DataLoader(val_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=0,   
        pin_memory=True)
    test_loader = DataLoader(test_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=0,   
        pin_memory=True)

    print(f"Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    # return train_loader, val_loader, test_loader
    return train_loader, train_loader, train_loader
    

'''
----------------- graph data loader with params ----------------
'''
def compute_global_descriptor_vector(point_cloud: torch.Tensor) -> torch.Tensor:
    """
    Compute a global descriptor vector from a point cloud (M, 3).

    Output: Tensor of shape (15,)
        [centroid(3), bbox_size(3), pca_eigenvalues(3), 
         moment_trace(1), moment_det(1), moment_frobenius(1),
         mean_distance_to_centroid(1), bbox_min(3), bbox_max(3)]
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "Input must be of shape (M, 3)"
    M = point_cloud.shape[0]

    # Centroid
    centroid = point_cloud.mean(dim=0)  # (3,)

    # Bounding box
    min_xyz = point_cloud.min(dim=0).values  # (3,)
    max_xyz = point_cloud.max(dim=0).values  # (3,)

    # Centered cloud for PCA
    centered = point_cloud - centroid
    cov = centered.T @ centered / M
    eigenvalues, _ = torch.linalg.eigh(cov)  # (3,), ascending
    pca_eigenvalues = eigenvalues.flip(0)   # descending

    # Concatenate all features into one vector
    descriptor = torch.cat([
        centroid,           # 3
        pca_eigenvalues,    # 3
        min_xyz,            # 3
        max_xyz             # 3
    ])  # total: 3 + 3 + 3 + 1 + 1 + 1 + 1 + 3 + 3 = 19

    return descriptor  # (12,)

def point_cloud_to_density_grid(point_cloud: torch.Tensor, grid_resolution=32):
    """
    Convert a point cloud (M, 3) into a 3D density grid.
    
    Args:
        point_cloud (torch.Tensor): Input tensor of shape (M, 3)
        grid_resolution (int): Number of bins along each axis

    Returns:
        density_grid (torch.Tensor): Tensor of shape (grid_resolution, grid_resolution, grid_resolution)
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "Input must be (M, 3)"
    
    # Normalize the point cloud into [0, 1]^3
    min_xyz = point_cloud.min(dim=0).values
    max_xyz = point_cloud.max(dim=0).values
    normalized_pc = (point_cloud - min_xyz) / (max_xyz - min_xyz + 1e-6)

    # Scale to grid coordinates
    coords = (normalized_pc * grid_resolution).long()
    coords = torch.clamp(coords, 0, grid_resolution - 1)

    # Initialize the grid
    density_grid = torch.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=torch.float32)

    # Count points in each voxel
    for i in range(point_cloud.shape[0]):
        x, y, z = coords[i]
        density_grid[x, y, z] += 1.0

    return density_grid / torch.sum(density_grid)

class GraphDataset_self_param(Dataset):
    def __init__(self, graph_paths, param_type='high_level'):
        super(GraphDataset_self_param, self).__init__()

        # Graph file paths
        self.graph_paths = graph_paths
        self.param_type = param_type

    def __len__(self):
        """Returns the number of graphs available in the dataset."""
        return len(self.graph_paths)
    
    def len(self): 
        return len(self.graph_paths)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        """Loads a graph from file and optionally returns its grid shape."""
        # single_graph_path = self.graph_paths[idx]
        # graph = torch.load(single_graph_path, weights_only=False)

        single_graph_path = self.graph_paths[idx]
        try:
            graph = torch.load(single_graph_path, weights_only=False)
        except Exception as e:
            print(f"Failed to load {single_graph_path}: {e}")
            # Option 1: skip or return dummy data
            return self.__getitem__((idx + 1) % len(self.graph_paths))

        # add the params
        all_coors = graph.x
        if self.param_type == 'high_level3':
            params = compute_global_descriptor_vector(all_coors)
            graph.param = params
        elif self.param_type == 'grid':
            st = time.time()
            grid_rep = point_cloud_to_density_grid(all_coors, grid_resolution=64)    # 0.5s
            graph.grid_rep = grid_rep.unsqueeze(0).unsqueeze(0)    # (1, 1, 64, 64, 64)
        
        return graph

def create_data_loaders_with_param(graph_folder_path, param_type='high_level', batch_size=1):
    """
    Creates train, validation, and test data loaders.

    Parameters:
    - graph_paths (list): List of file paths to graph data.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - train_loader, val_loader, test_loader: DataLoader objects for each split.
    """
    graph_paths = get_graph_paths(graph_folder_path)
    num_samples = len(graph_paths)

    # Create dataset
    dataset = GraphDataset_self_param(graph_paths, param_type)

    # Define train, validation, and test split indices
    train_size = 2 # int(0.6 * num_samples)
    val_size = int(0.01 * num_samples)
    test_size = int(1 * num_samples) - train_size - val_size

    # Split dataset
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=True,
        num_workers=1,   
        pin_memory=False)
    val_loader = DataLoader(val_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=1,   
        pin_memory=True)
    test_loader = DataLoader(test_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=1,   
        pin_memory=True)

    print(f"Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    # return train_loader, val_loader, test_loader

    return train_loader, train_loader, train_loader





'''
--------------- geo data loader ----------------
'''
def create_geo_data_loaders(graph_folder_path, batch_size=1):
    """
    Creates train, validation, and test data loaders.

    Parameters:
    - graph_paths (list): List of file paths to graph data.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - train_loader, val_loader, test_loader: DataLoader objects for each split.
    """
    graph_paths = get_graph_paths(graph_folder_path)
    num_samples = len(graph_paths)

    # Create dataset
    dataset = GraphDataset(graph_paths)

    # Split dataset
    train_dataset = torch.utils.data.Subset(dataset, range(0, num_samples))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=False,
        num_workers=4,   
        pin_memory=True)

    return train_loader



