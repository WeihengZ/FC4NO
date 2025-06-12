import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Activation Dictionary ----------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class Geom_DeepONet(nn.Module):
    def __init__(self, args):
        super(Geom_DeepONet, self).__init__()

        self.act = ACTIVATION[args.act]()
        self.last_act = ACTIVATION[args.last_act]()
        self.out_dim = args.out_dim

        layer_sizes_branch = [
            [args.geo_param_dim+args.load_param_dim] + [args.hidden_dim for _ in range(args.num_layers-1)], 
            [args.hidden_dim for _ in range(args.num_layers)]
        ]
        layer_sizes_trunk = [
            [args.coor_dim] + [args.hidden_dim for _ in range(args.num_layers-1)], 
            [args.hidden_dim for _ in range(args.num_layers-1)] + [args.out_dim * args.hidden_dim]
        ]

        # Branch network (geoNet and geoNet2)
        self.geoNet = self.build_branch_net(layer_sizes_branch[0])
        self.geoNet2 = self.build_branch_net(layer_sizes_branch[1])
        
        # Trunk network (outNet and outNet2)
        self.outNet = self.build_trunk_net(layer_sizes_trunk[0])
        self.outNet2 = self.build_trunk_net(layer_sizes_trunk[1])
        
        self.b = nn.Parameter(torch.zeros(self.out_dim))  # Bias term
    
    def build_branch_net(self, layer_sizes):
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(self.act)  # Activation
        return nn.Sequential(*layers)

    def build_trunk_net(self, layer_sizes):
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(self.act)  # Activation
        return nn.Sequential(*layers)

    def forward(self, graph):

        assert hasattr(graph, 'params') == True, 'Geom-DeepONet can only handle data with parametric representation'

        x_func1 = graph.params.unsqueeze(0)  # Input geometry (size [bs, param_dim])
        x_loc = graph.x.unsqueeze(0)   # Coordinates (size [bs, Npt, coor_dim])
        
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
        x_loc = self.outNet2(mix)  # Output: [bs, Npt, hidden_dim * c]
        
        # Expand dimensions for final computation
        B, NN, _ = x_loc.shape
        x_loc = x_loc.reshape(B, NN, -1, self.out_dim)  # Output: [bs, Npt, hidden_dim, c]
        
        # Final operation
        x = torch.einsum("bh,bnhc->bnc", x_func1, x_loc)  # Output: [bs, Npt, c]
        
        # Add bias
        x += self.b
        
        # Sigmoid activation
        return self.last_act(x).squeeze(0)