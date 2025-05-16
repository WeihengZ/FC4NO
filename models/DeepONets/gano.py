import torch.nn as nn
import torch

class GANO(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim, out_dim):
        super(GANO, self).__init__()

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


