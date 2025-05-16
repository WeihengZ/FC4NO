import torch.nn as nn
import torch


class DCON(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super(DCON, self).__init__()

        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.FC1u = nn.Linear(3, hidden_dim)
        self.FC2u = nn.Linear(hidden_dim, hidden_dim)
        self.FC3u = nn.Linear(hidden_dim, hidden_dim)
        self.FC4u = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, graph):
        '''
        data.x   (M, 3)
        loading  (1, 101)
        '''
        assert hasattr(graph, 'param') == True, 'DCON can only handle data with parametric representation'

        # compute loading embedding
        enc = self.branch(graph.param)    # (1, F)

        # extract graph data
        x = graph.x
        u = self.FC1u(x)    # (M, F)

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