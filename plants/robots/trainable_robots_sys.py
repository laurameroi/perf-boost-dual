import torch
import torch.nn.functional as F

from . import RobotsSystem

class TrainableRobotsSystem(RobotsSystem):
    def __init__(
        self, x_target: torch.Tensor, 
        init_k: float, init_b: float, init_mass: float,
        linear_plant: bool, init_b2: float=None, x_init=None, u_init=None
    ):
        """

        Args:
            x_target: concatenated nominal initial point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to x_target when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super(RobotsSystem, self).__init__()

        self.linear_plant = linear_plant
        self.tanh_nonlinearity = False

        # initial state
        self.register_buffer('x_target', x_target.reshape(1, -1))  # shape = (1, state_dim)
        x_init = self.x_target.detach().clone() if x_init is None else x_init.reshape(1, -1)   # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.x_target.shape[1]/2)) if u_init is None else u_init.reshape(1, -1)   # shape = (1, input_dim)
        self.register_buffer('u_init', u_init)
        # check dimensions
        self.n_agents = int(self.x_target.shape[1]/4)
        self.state_dim = 4*self.n_agents
        self.input_dim = 2*self.n_agents
        self.output_dim = self.state_dim #TODO
        assert self.x_target.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.input_dim

        if self.linear_plant:
            assert init_b2 is None

        self.h = 0.05
        self.mass = torch.nn.Parameter(torch.tensor(init_mass)) # true value is 1.0
        self.k = torch.nn.Parameter(torch.tensor(init_k))       # true value is 1.0
        self.b = torch.nn.Parameter(torch.tensor(init_b))       # true value is 1.0
        self.b2 = None if self.linear_plant else torch.nn.Parameter(torch.tensor(init_b2))  # true value is 0.1

        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)
        self.register_buffer('mask', mask)

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, input_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        # define A and B
        b_temp = torch.zeros(4,2)
        b_temp[2, 0] = 1/self.mass
        b_temp[3, 1] = 1/self.mass
        B = torch.kron(torch.eye(self.n_agents), b_temp) * self.h
        B = B.to(x.device)

        _A1 = torch.eye(4*self.n_agents)
        mat1 = torch.zeros(2,2)
        mat1[0,0] = -self.k/self.mass 
        mat1[1,1] = -self.k/self.mass
        mat2 = torch.zeros(2,2)
        mat2[0,0] = -self.b/self.mass 
        mat2[1,1] = -self.b/self.mass
        _A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                torch.eye(2)
                                ), dim=1),
                        torch.cat((mat1, mat2), dim=1),
                        ), dim=0)
        _A2 = torch.kron(torch.eye(self.n_agents), _A2)
        A_lin = _A1 + self.h * _A2
        A_lin = A_lin.to(x.device)

        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.input_dim)
        if self.linear_plant:
            # x is batched but A is not => can use F.linear to compute xA^T
            f = F.linear(x - self.x_target, A_lin) + F.linear(u, B) + self.x_target
        else:
            if not self.tanh_nonlinearity:
                # A depends on x, hence is batched. perform batched matrix multiplication
                f = torch.bmm(x - self.x_target, self.A_nonlin(x, A_lin).transpose(1,2)) + F.linear(u, B) + self.x_target
            else:
                f = (F.linear(x - self.x_target, A_lin)
                     + self.h * self.b2 / self.mass * self.mask.view(-1) * torch.tanh(x - self.x_target)
                     + F.linear(u, B) + self.x_target)
        return f    # shape = (batch_size, 1, state_dim)
    
    def A_nonlin(self, x, A_lin):
        assert not self.linear_plant
        A3 = torch.norm(
            x.view(-1, 2 * self.n_agents, 2) * self.mask, dim=-1, keepdim=True
        )           # shape = (batch_size, 2 * n_agents, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1, device=A3.device)
        )           # shape = (batch_size, 4 * n_agents, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        )           # shape = (batch_size, 4 * n_agents, 4 * n_agents)
        A = A_lin + self.h * A3
        return A    # shape = (batch_size, 4 * n_agents, 4 * n_agents)
