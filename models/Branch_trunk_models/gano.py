import torch.nn as nn
import torch

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
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

class GANO(nn.Module):
    def __init__(self, args):
        super(GANO, self).__init__()

        self.act = ACTIVATION[args.act]()
        self.last_act = ACTIVATION[args.last_act]()
        self.varying_parametric_geometry = False
        self.varying_pointcloud_geometry = False
        self.varying_loading = False

        # geo encoder
        if args.geo_param_dim > 0:
            self.varying_parametric_geometry = True
            self.parametric_geo_branch = MLP(n_input=args.geo_param_dim, 
                n_hidden=args.hidden_dim, n_output=args.hidden_dim, 
                n_layers=args.num_layers, act=args.act, res=False)
        else:
            self.varying_pointcloud_geometry = True
            self.point_based_geo_branch = MLP(n_input=args.coor_dim,
                n_hidden=args.hidden_dim, n_output=args.hidden_dim, 
                n_layers=args.num_layers, act=args.act, res=False)
        
        # loading encoder
        if args.load_param_dim > 0:
            self.varying_loading = True
            self.loading_geo_branch = MLP(n_input=args.load_param_dim, 
                n_hidden=args.hidden_dim, n_output=args.hidden_dim, 
                n_layers=args.num_layers, act=args.act, res=False)
        
        # define local_global combination layers
        self.local_global_branch = MLP(n_input=args.hidden_dim * 2, 
            n_hidden=args.hidden_dim, n_output=args.hidden_dim, 
            n_layers=args.num_layers, act=args.act, res=False)

        # define trunk layer
        self.trunks = [nn.Linear(args.coor_dim, args.hidden_dim)]
        for _ in range(args.num_layers):
            self.trunks.append(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.trunks.append(nn.Linear(args.hidden_dim, args.out_dim))
        self.trunks = nn.ModuleList(self.trunks)

    def forward(self, graph):
        '''
        data.x   (M, 3)
        '''

        # extract graph data coordinate
        x = graph.x
        N = x.shape[0]

        # extract geometry embedding
        if self.varying_parametric_geometry:
            geo_params = self.parametric_geo_branch(graph.geo_params).unsqueeze(0).repeat(N, 1)
        elif self.varying_pointcloud_geometry:
            geo_params = torch.amax(self.point_based_geo_branch(x), 0, keepdim=True).repeat(N, 1)
        
        # extract loading embeddings
        if self.varying_loading:
            enc = self.loading_geo_branch(graph.load_params)

        # forward
        u = self.trunks[0](x)    # (M, F)
        u = torch.cat((u, geo_params), dim=-1)    # (M, F+F)

        u = self.local_global_branch(u)    # (M, F)

        # operator layers
        for j in range(1, len(self.trunks)):
            u = self.act(u)
            if self.varying_loading:
                u = u * enc
            u = self.trunks[j](u)

        return self.last_act(u)


