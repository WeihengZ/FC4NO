import torch.nn as nn
import torch

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class DCON(nn.Module):
    def __init__(self, args):
        super(DCON, self).__init__()

        self.act = ACTIVATION[args.act]()
        self.last_act = ACTIVATION[args.last_act]()

        # define branch layer
        branch_layers = [nn.Linear(args.geo_param_dim+args.load_param_dim, args.hidden_dim), self.act]
        branch_layers = branch_layers + [
            nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim), self.act) 
        for _ in range(args.num_layers)]
        self.branch = nn.Sequential(*branch_layers)

        # define trunk layer
        self.trunks = [nn.Linear(args.coor_dim, args.hidden_dim)]
        for _ in range(args.num_layers):
            self.trunks.append(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.trunks.append(nn.Linear(args.hidden_dim, args.out_dim))
        self.trunks = nn.ModuleList(self.trunks)

    def forward(self, graph):
        '''
        data.x   (M, 3)
        loading  (1, 101)
        '''
        assert hasattr(graph, 'params') == True, 'DCON can only handle data with parametric representation'

        # compute loading embedding
        enc = self.branch(graph.params)    # (1, F)

        # extract graph data
        x = graph.x
        u = self.trunks[0](x)    # (M, F)

        # operator layers
        for j in range(1, len(self.trunks)):
            u = self.act(u)
            u = u * enc
            u = self.trunks[j](u)

        return self.last_act(u)