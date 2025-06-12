import torch.nn as nn
import torch
import numpy as np

# ---------- Activation Dictionary ----------
ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)])
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.res = res

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)

class DON(nn.Module):
    def __init__(self, args):
        super(DON, self).__init__()

        self.branch = MLP(n_input=args.geo_param_dim+args.load_param_dim, 
            n_hidden=args.hidden_dim, n_output=args.out_dim * args.hidden_dim, 
            n_layers=args.num_layers, act=args.act, res=False)
        self.trunk = MLP(n_input=args.coor_dim, 
            n_hidden=args.hidden_dim, n_output=args.out_dim * args.hidden_dim, 
            n_layers=args.num_layers, act=args.act, res=False)
        self.last_act = ACTIVATION[args.last_act]()
        self.out_dim = args.out_dim

    def forward(self, graph):
        '''
        graph.x:    (B, M, trunk_dim)
        pde_param:  (B, branch_dim)
        '''
        '''
        This model only work for the dataset with param
        '''   
        assert hasattr(graph, 'params'), 'DeepONet only works for the dataset with parametric representations'
        params = graph.params.unsqueeze(0)    # (B, branch_dim)

        # compute branch embedding
        pde_param = self.branch(params).reshape(1, -1, self.out_dim).unsqueeze(1)    # (B, 1, F, out_dim)

        # compute trunk embedding
        x = graph.x    # (B, M, trunk_dim)

        num_nodes, _ = x.shape
        x = self.trunk(x).reshape(1, num_nodes, -1, self.out_dim)    # (B, M, F, out_dim)

        # final output
        out = torch.mean(x * pde_param, -2).squeeze(0)    # (M, out_dim)

        return self.last_act(out)
