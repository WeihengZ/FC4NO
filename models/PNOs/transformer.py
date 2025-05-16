import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
import numpy as np

class MLP(nn.Module):
    def __init__(self, width: int, in_channels: Optional[int] = None, out_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = width
        if out_channels is None:
            out_channels = width
        self.width = width
        self.c_fc = nn.Linear(in_channels, width * 4)
        self.c_proj = nn.Linear(width * 4, out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)
        self.mlp = MLP(width=embed_dim)
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor):
        x = x + self.attn(self.ln_q(x), self.ln_k(kv), self.ln_v(kv))[0]
        x = x + self.mlp(self.ln_out(x))
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)
        self.mlp = MLP(width=embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_q(x), self.ln_k(x), self.ln_v(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Transformer, self).__init__()

        self.coor_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.kv_coor_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # attention layer
        self.cross_attn1 = ResidualCrossAttentionBlock(embed_dim=hidden_channels, num_heads=8)
        self.cross_attn2 = ResidualCrossAttentionBlock(embed_dim=hidden_channels, num_heads=8)
        self.cross_attn3 = ResidualCrossAttentionBlock(embed_dim=hidden_channels, num_heads=8)
        self.gelu = nn.GELU()

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, graph):
        '''
        transformer has huge memory requirement
        Hence, for the query point, we can only do random sampling during the training
        '''
        assert hasattr(graph, 'control_points') == True, 'transformer need point cloud of geometry representation'
        # add the parameter features in if possible

        # sampling
        num_trunks = int(graph.x.shape[0] / 10000) + 1
        trunk_id = np.random.randint(num_trunks)

        # extract data
        x = torch.chunk(graph.x, num_trunks, 0)[trunk_id]
        if hasattr(graph, 'param') == True:
            x = torch.cat((x, graph.param.unsqueeze(0).repeat(x.shape[0], 1)), -1)

        # compute embedding
        x = self.coor_encoder(x).unsqueeze(0)    # (1, M, F)
        kv = self.kv_coor_encoder(graph.control_points).unsqueeze(0)

        # attention layer
        x = self.cross_attn1(x, kv)
        x = self.cross_attn2(x, kv)
        x = self.cross_attn3(x, kv)
        
        # compute the global features
        xg = x.squeeze(0)

        # final prediction
        out =  self.decoder(xg)

        # create ground truth
        y_gt = torch.chunk(graph.y, num_trunks, 0)[trunk_id]

        return out.squeeze(-1), y_gt.squeeze(-1)
    
    def predict(self, graph):
        '''
        seperate the inference for memory limit
        '''
        assert hasattr(graph, 'control_points') == True, 'transformer need point cloud of geometry representation'
        # add the parameter features in if possible
        x = graph.x
        if hasattr(graph, 'param') == True:
            x = torch.cat((x, graph.param.unsqueeze(0).repeat(x.shape[0], 1)), -1)

        # extract point data
        x = self.coor_encoder(x).unsqueeze(0)    # (1, M, F)
        kv = self.kv_coor_encoder(graph.control_points).unsqueeze(0)

        # chunk all the coordinates
        num_trunks = int(graph.x.shape[0] / 10000) + 1
        all_preds = []
        x_chunk = torch.chunk(x, num_trunks, 1)
        for j in range(num_trunks):

            # attention layer
            x = self.cross_attn1(x_chunk[j], kv)
            x = self.cross_attn2(x, kv)
            x = self.cross_attn3(x, kv)
            
            # compute the global features
            xg = x.squeeze(0)

            # final prediction
            out = self.decoder(xg)
            out = out.detach().cpu().numpy()
            all_preds.append(out)
        all_preds = np.concatenate(tuple(all_preds), 0)

        return all_preds