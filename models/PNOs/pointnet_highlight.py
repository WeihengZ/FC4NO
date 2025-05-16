import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet_highlight(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, additional_features=None):
        super(PointNet_highlight, self).__init__()

        # Coordinate Encoder using linear layers (instead of convolutions)
        self.coor_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Global Feature Aggregation (using max pooling or other aggregation methods)
        self.global_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, 2*hidden_channels)
        )

        # Fully connected layers for prediction (decoder)
        self.decoder = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )
        
        self.additional_features = additional_features
    
    def forward(self, graph):
        '''
        graph.x contains the point cloud data (each point in the cloud with features)
        '''
        # add the parameter features in if possible
        x = graph.x
        if hasattr(graph, 'param') == True:
            x = torch.cat((x, graph.param.unsqueeze(0).repeat(x.shape[0], 1)), -1)
            
        # Extract point cloud features (assuming graph.x shape is [num_points, in_channels])
        x = x.unsqueeze(0)  # Add batch dimension (1, num_points, in_channels)

        # Pass through coordinate encoder (apply on each point independently)
        x = self.coor_encoder(x)  # (1, num_points, hidden_channels)

        # Aggregate features across all points (global feature)
        global_features = torch.max(self.global_feature(x), dim=1)[0]  # (1, hidden_channels)
        global_features = global_features.unsqueeze(1).repeat(1,x.shape[1],1)
        xg = torch.cat((x, global_features), -1)
        
        # Pass through decoder
        out = self.decoder(xg)  # (1, num_points, out_channels)

        return out.squeeze(0)  # Remove batch dimension, shape (out_channels)
