import torch.nn as nn
import torch

ACTIVATION = {
    'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
    'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,
    'identity': nn.Identity
}

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

class S_DeepONet(nn.Module):
    def __init__(self, args, input_fn_dim=1):
        """
        PyTorch implementation of DeepONet with Cartesian Product.

        Args:
        - m (int): Sequence length for the branch network.
        - hidden_dim (int): Number of hidden units in the branch and trunk networks.
        - n_components (int): Output components (e.g., temperature, stress).
        - input_fn_dim (int): Dimension of the input function (default=1).
        """
        super(S_DeepONet, self).__init__()

        self.n_components = args.out_dim

        # === Branch Network (GRU) ===
        self.branch = BranchNet(hidden_size=args.hidden_dim, output_size=args.out_dim, N_input_fn=input_fn_dim)
        act = ACTIVATION[args.act]()
        self.last_act = ACTIVATION[args.last_act]()

        # === Trunk Network (MLP) ===
        self.trunk = []
        self.trunk.append(nn.Linear(args.coor_dim, args.hidden_dim))
        self.trunk.append(act)
        for _ in range(args.num_layers):
            self.trunk.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            self.trunk.append(act)
        self.trunk.append(nn.Linear(args.hidden_dim, (args.load_param_dim) * args.out_dim))
        self.trunk = nn.Sequential(*self.trunk)

        # Learnable bias term
        self.bias = nn.Parameter(torch.zeros(self.n_components))

    def forward(self, graph):
        '''
        graph.x:    (B, M, trunk_dim)
        pde_param:  (B, branch_dim)
        '''

        assert hasattr(graph, 'time_dependent_params') == True, 'S-DeepONet can only handle data with parametric representation'

        x_loc = graph.x
        num_nodes = x_loc.shape[0]
        x_func = graph.time_dependent_params.unsqueeze(-1).unsqueeze(0)

        # === Branch Forward Pass ===
        x_func = self.branch(x_func)  # (1, T, 2)

        # === Trunk Forward Pass ===
        x_loc = self.trunk(x_loc)  # Shape: (M, 2T)
        x_loc = x_loc.view(num_nodes, -1, self.n_components)  # (M, T, 2)

        # === Dot Product ===
        # Compute einsum: "bh,nhc -> bnc"
        x = torch.mean(x_func * x_loc, 1)    # (M, 2)

        # Add bias and apply activation (sigmoid in original TF model)
        x += self.bias.unsqueeze(0)

        return self.last_act(x)