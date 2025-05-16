import torch
import torch.nn as nn
import torch.nn.functional as F

class Geom_DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim, activation='swish'):
        super(Geom_DeepONet, self).__init__()

        layer_sizes_branch = [[branch_dim, hidden_dim, hidden_dim], [hidden_dim, hidden_dim, hidden_dim]]
        layer_sizes_trunk = [[trunk_dim, hidden_dim, hidden_dim], [hidden_dim, hidden_dim, hidden_dim]]
        
        # Branch network (geoNet and geoNet2)
        self.geoNet = self.build_branch_net(layer_sizes_branch[0])
        self.geoNet2 = self.build_branch_net(layer_sizes_branch[1])
        
        # Trunk network (outNet and outNet2)
        self.outNet = self.build_trunk_net(layer_sizes_trunk[0])
        self.outNet2 = self.build_trunk_net(layer_sizes_trunk[1])
        
        self.b = nn.Parameter(torch.zeros(1))  # Bias term
    
    def build_branch_net(self, layer_sizes):
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Tanh())  # Activation
        return nn.Sequential(*layers)

    def build_trunk_net(self, layer_sizes):
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Tanh())  # Activation
        return nn.Sequential(*layers)

    def forward(self, graph):

        assert hasattr(graph, 'param') == True, 'Geom-DeepONet can only handle data with parametric representation'

        x_func1 = graph.param.unsqueeze(0)  # Input geometry (size [bs, N])
        x_loc = graph.x.unsqueeze(0)   # Coordinates (size [bs, Npt, 4])
        
        # Encode implicit geometry
        x_func1_1 = self.geoNet(x_func1)  # Output: [bs, hidden_dim]
        
        # Encode output coordinates
        x_loc_1 = self.outNet(x_loc)  # Output: [bs, Npt, hidden_dim]

        # Mix data
        mix = torch.einsum("bh,bnh->bnh", x_func1_1, x_loc_1)  # Output: [bs, Npt, hidden_dim]
        
        # Reduce across the Npt dimension
        mix_reduced = torch.mean(mix, dim=1)  # Output: [bs, hidden_dim]
        
        # Further encode
        x_func1 = self.geoNet2(mix_reduced)  # Output: [bs, hidden_dim]
        x_loc = self.outNet2(mix)  # Output: [bs, Npt, hidden_dim]
        
        # Expand dimensions for final computation
        x_loc = x_loc.unsqueeze(-1)  # Output: [bs, Npt, hidden_dim, 1]
        
        # Final operation
        x = torch.einsum("bh,bnhc->bnc", x_func1, x_loc)  # Output: [bs, Npt, 1]
        
        # Add bias
        x += self.b
        
        # Sigmoid activation
        return torch.sigmoid(x).squeeze(0)