import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()

        self.act = ACTIVATION[args.act]
        self.last_act = ACTIVATION[args.last_act]()
        in_channels = args.coor_dim
        hidden_channels = args.hidden_dim
        out_channels = args.out_dim

        self.coor_encoder = []
        self.coor_encoder.append(nn.Linear(in_channels, hidden_channels))
        self.coor_encoder.append(self.act())
        for i in range(args.num_layers-1):
            self.coor_encoder.append(nn.Linear(hidden_channels, hidden_channels))
            self.coor_encoder.append(self.act())
        self.coor_encoder.append(nn.Linear(hidden_channels, hidden_channels))
        self.coor_encoder = nn.ModuleList(self.coor_encoder)

        self.global_feature = []
        for i in range(args.num_layers):
            self.global_feature.append(nn.Linear(hidden_channels, hidden_channels))
            self.global_feature.append(self.act())
        self.global_feature.append(nn.Linear(hidden_channels, 3*hidden_channels))
        self.global_feature = nn.ModuleList(self.global_feature)

        self.decoder = []
        self.decoder.append(nn.Linear(4*hidden_channels, hidden_channels))
        self.decoder.append(self.act())
        for i in range(args.num_layers-1):
            self.decoder.append(nn.Linear(hidden_channels, hidden_channels))
            self.decoder.append(self.act())
        self.decoder.append(nn.Linear(hidden_channels, out_channels))
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, graph):
        '''
        graph.x contains the point cloud data (each point in the cloud with features)
        '''
        # add the parameter features in if possible
        x = graph.x
            
        # Extract point cloud features (assuming graph.x shape is [num_points, in_channels])
        x = x.unsqueeze(0)  # Add batch dimension (1, num_points, in_channels)

        # Pass through coordinate encoder (apply on each point independently)
        for layer in self.coor_encoder:
            x = layer(x)

        # Aggregate features across all points (global feature)
        for layer in self.global_feature:
            xg = layer(x)
        global_features = torch.max(xg, dim=1)[0]  # (1, 3*hidden_channels)
        global_features = global_features.unsqueeze(1).repeat(1,x.shape[1],1)
        xlg = torch.cat((x, global_features), -1)
        
        # Pass through decoder
        for layer in self.decoder:
            xlg = layer(xlg)
        out = xlg  # (1, num_points, out_channels)

        return self.last_act(out.squeeze(0))  # Remove batch dimension, shape (out_channels)

