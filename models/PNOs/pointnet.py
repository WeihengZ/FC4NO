import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, additional_features=None):
        super(PointNet, self).__init__()

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

''' ------------------------------------------------- '''
class BranchNet(nn.Module):
    def __init__(self, hidden_size=256, output_size=1, N_input_fn=1):
        super(BranchNet, self).__init__()
        self.gru1 = nn.GRU(input_size=N_input_fn, hidden_size=hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.repeat_vector = nn.Linear(hidden_size, hidden_size)  # Equivalent to RepeatVector(HIDDEN)
        self.gru3 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.gru4 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # TimeDistributed Dense in Keras

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.repeat_vector(x[:, -1, :]).unsqueeze(1).repeat(1, x.shape[1], 1)
        x, _ = self.gru3(x)
        x, _ = self.gru4(x)
        x = self.fc(x)

        return x  # Shape: [batch_size, m, output_size]

class S_PointNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, additional_features=None):
        super(S_PointNet, self).__init__()

        self.branch = BranchNet(hidden_size=hidden_channels, output_size=out_channels)

        # Coordinate Encoder using linear layers (instead of convolutions)
        self.coor_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.mlp = nn.Sequential(
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
            pde_param = graph.param.unsqueeze(0).unsqueeze(-1)
            pde_param = self.branch(pde_param).squeeze(-1).unsqueeze(1)
            
        # Extract point cloud features (assuming graph.x shape is [num_points, in_channels])
        x = x.unsqueeze(0)  # Add batch dimension (1, num_points, in_channels)

        # Pass through coordinate encoder (apply on each point independently)
        x = self.coor_encoder(x)  # (1, num_points, hidden_channels)

        # Aggregate features across all points (global feature)
        global_features = torch.max(self.global_feature(x), dim=1)[0]  # (1, hidden_channels)
        global_features = global_features.unsqueeze(1).repeat(1,x.shape[1],1)
        xg = torch.cat((x, global_features), -1)
        
        # Pass through decoder
        xg = xg * pde_param
        out = self.decoder(xg)  # (1, num_points, out_channels)

        return torch.sigmoid(out.squeeze(0))  # Remove batch dimension, shape (out_channels)
