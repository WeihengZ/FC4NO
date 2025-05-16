import torch.nn as nn
import torch
import numpy as np

class DON(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super(DON, self).__init__()

        self.branch = nn.Sequential(
            nn.Linear(branch_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim *hidden_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim *hidden_dim)
        )
        self.out_dim = out_dim

    def forward(self, graph):

        '''
        graph.x:    (B, M, trunk_dim)
        pde_param:  (B, branch_dim)
        '''

        '''
        This model only work for the dataset with param
        '''
        assert hasattr(graph, 'param') == True, 'DeepONet can only handle data with parametric representation'

        # extract nodal cordinates
        x = graph.x
        pde_param = graph.param
        num_nodes, _ = x.shape
        
        # compute branch embedding
        pde_param = self.branch(pde_param).reshape(1, -1, self.out_dim).unsqueeze(1)    # (B, 1, F, out_dim)

        # compute trunk embedding
        x = self.trunk(x).reshape(1, num_nodes, -1, self.out_dim)    # (B, M, F, out_dim)

        # final output
        out = torch.mean(x * pde_param, -2).squeeze(0)    # (M, out_dim)

        return torch.sigmoid(out)

class DON_cnn32(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super(DON_cnn32, self).__init__()

        self.branch = nn.Sequential(
            nn.Linear(branch_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        
        self.FC2u = nn.Linear(hidden_dim, hidden_dim)
        self.FC3u = nn.Linear(hidden_dim, hidden_dim)
        self.FC4u = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, graph):

        '''
        graph.x:    (M, trunk_dim)
        pde_param:  (branch_dim)
        '''
        
        if hasattr(graph, 'param'):

            # extract nodal cordinates
            x = graph.x
            enc = self.branch(graph.param)    # (1, F)

            # compute trunk embedding
            u = self.trunk(x)    # (M, F)

            # operator layers
            #
            u = self.act(u)
            u = u * enc
            u = self.FC2u(u) 
            # 
            u = self.act(u)
            u = u * enc
            u = self.FC3u(u) 
            #  
            u = self.act(u)
            u = u * enc
            out = self.FC4u(u)

        else:

            # extract nodal cordinates
            x = graph.x
            control_points = graph.control_points
            num_nodes, _ = x.shape

            # compute parameter embedding
            enc = self.branch(control_points)    # (M, F)
            enc = torch.amax(enc, 0, keepdims=True)    # (1, F)
            u = self.trunk(x)    # (M, F)

            # operator layers
            #
            u = self.act(u)
            u = u * enc
            u = self.FC2u(u) 
            # 
            u = self.act(u)
            u = u * enc
            u = self.FC3u(u) 
            #  
            u = self.act(u)
            u = u * enc
            out = self.FC4u(u)

        return torch.sigmoid(out)



class DON_cnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super(DON_cnn, self).__init__()
        self.M = 10
        self.N = 10
        self.K = 10

        self.total_branch = nn.Sequential(
                nn.Linear(branch_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim *hidden_dim)
            )

        self.branchs = nn.ModuleList()
        for i in range(3):
            self.branchs.append(nn.Sequential(
                nn.Linear(branch_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim *hidden_dim)
            ))
        self.trunks = nn.ModuleList()
        for i in range(3):
            self.trunks.append(nn.Sequential(
                nn.Linear(trunk_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim *hidden_dim)
            ))
        self.outs = nn.ModuleList()
        for i in range(3):
            self.outs.append(nn.Sequential(
                nn.Linear(3*hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, out_dim)
            ))
        self.out_dim = out_dim
    
    def divide_into_patches(self, graph):
        """
        Partition graph.x and graph.y into spatial patches based on [0, 1] coordinates.

        Returns:
            x_patch_list: list of tensors, coordinates in each patch
            y_patch_list: list of tensors, ground truth values in each patch
        """
        coords = graph.x  # (N, 3)
        values = graph.y  # (N, out_dim)

        # Compute bin index directly from [0,1] scaled coordinates
        x_bin = torch.floor(coords[:, 0] * self.M).long()
        y_bin = torch.floor(coords[:, 1] * self.N).long()
        z_bin = torch.floor(coords[:, 2] * self.K).long()

        # Clamp to ensure indices are within range [0, M/N/K - 1]
        x_bin = x_bin.clamp(0, self.M - 1)
        y_bin = y_bin.clamp(0, self.N - 1)
        z_bin = z_bin.clamp(0, self.K - 1)

        patch_ids = x_bin * self.N * self.K + y_bin * self.K + z_bin  # (N,)

        # Initialize empty lists
        x_patch_list = [[] for _ in range(self.M * self.N * self.K)]
        y_patch_list = [[] for _ in range(self.M * self.N * self.K)]

        for i in range(coords.shape[0]):
            idx = patch_ids[i].item()
            x_patch_list[idx].append(coords[i])
            y_patch_list[idx].append(values[i])

        # Convert to tensors
        x_patch_list = [torch.stack(patch) if patch else torch.empty((0, 3), device=coords.device) for patch in x_patch_list]
        y_patch_list = [torch.stack(patch) if patch else torch.empty((0, values.shape[1]), device=values.device) for patch in y_patch_list]

        return x_patch_list, y_patch_list
    
    def compute_patch_centroids(self, x_patch_list):
        """
        Compute the mean coordinate (centroid) for each patch.

        Returns:
            centroid_vector: (3 * M * N * K,) tensor
        """
        centroids = []
        for patch in x_patch_list:
            if patch.shape[0] > 0:
                centroids.append(torch.mean(patch, dim=0))  # (3,)
            else:
                centroids.append(torch.zeros(3, device=patch.device))  # if empty, fill with 0s

        centroid_vector = torch.cat(centroids, dim=0)  # (3 * M * N * K,)
        return centroid_vector

    def forward(self, graph):

        '''
        graph.x:    (B, M, trunk_dim)
        '''

        global_param = torch.amax(self.total_branch(graph.x), 0).unsqueeze(0).unsqueeze(0)

        x_patch_list, y_patch_list = self.divide_into_patches(graph)

        # find most difficult patch
        max_vals = [patch.max().item() if patch.numel() > 0 else float('-inf') for patch in y_patch_list]
        max_idx = int(torch.tensor(max_vals).argmax())
        non_void_patch_ids = [i for i, x in enumerate(x_patch_list) if x.shape[0] > 0]
        first_idx = np.amin(non_void_patch_ids)

        # compute pde solution for one signle patch
        patch_id = 995
        x_patch = x_patch_list[patch_id]
        gt_patch = y_patch_list[patch_id]

        # model forward
        local_param = torch.amax(self.branchs[0](graph.x), 0).unsqueeze(0).unsqueeze(0)
        local_coord_param = self.trunks[0](x_patch).unsqueeze(0)
        MM = local_coord_param.shape[1]

        pred_patch = torch.cat((local_coord_param, local_param.repeat(1,MM,1), global_param.repeat(1,MM,1)), -1)
        pred_patch = self.outs[0](pred_patch).squeeze(0)

        return pred_patch, gt_patch


# class DON_cnn2(nn.Module):
#     def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
#         super(DON_cnn, self).__init__()

#         # branch_net with fixed resolution 64x64x64
#         # map to final hidden embedding of shape of 256
#         self.branch = nn.Sequential(
#             nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),  # -> (B, 4, 32, 32, 32)
#             nn.GELU(),
#             nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1), # -> (B, 8, 16, 16, 16)
#             nn.GELU(),
#             nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),# -> (B, 16, 8, 8, 8)
#             nn.GELU(),
#             nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),# -> (B, 32, 4, 4, 4)
#             nn.GELU(),
#             nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),# -> (B, 64, 2, 2, 2)
#             nn.GELU(),
#             nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),# -> (B, 128, 1, 1, 1)
#             nn.Flatten(),                                           # -> (B, 128)
#             nn.Linear(128, hidden_dim), nn.GELU(),
#             nn.Linear(hidden_dim,hidden_dim), nn.GELU(),
#             nn.Linear(hidden_dim,hidden_dim), nn.GELU(),
#             nn.Linear(hidden_dim, out_dim *hidden_dim)
#         )
#         self.trunk = nn.Sequential(
#             nn.Linear(trunk_dim,5*hidden_dim), nn.Tanh(),
#             nn.Linear(5*hidden_dim,5*hidden_dim), nn.Tanh(),
#             nn.Linear(5*hidden_dim,5*hidden_dim), nn.Tanh(),
#             nn.Linear(5*hidden_dim, out_dim *hidden_dim)
#         )
#         self.out_dim = out_dim

#     def forward(self, graph):

#         '''
#         graph.x:    (B, M, trunk_dim)
#         pde_param:  (B, branch_dim)
#         '''

#         '''
#         This model only work for the dataset with param
#         '''
#         assert hasattr(graph, 'grid_rep') == True, 'DeepONet can only handle data with grid representation'

#         # extract nodal cordinates
#         x = graph.x
#         pde_param = graph.grid_rep
#         num_nodes, _ = x.shape
        
#         # compute branch embedding
#         pde_param = self.branch(pde_param).reshape(1, -1, self.out_dim).unsqueeze(1)    # (B, 1, F, out_dim)

#         # compute trunk embedding
#         x = self.trunk(x).reshape(1, num_nodes, -1, self.out_dim)    # (B, M, F, out_dim)

#         # final output
#         out = torch.mean(x, -2).squeeze(0)    # (M, out_dim)

#         return out # torch.sigmoid(out)
     