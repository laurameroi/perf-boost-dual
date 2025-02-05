import torch
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, x_target: torch.Tensor, linear_plant: bool, x_init, u_init=None, k: float=1.0):
        """

        Args:
            x_target: concatenated target point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to x_target when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        torch.manual_seed(0) #TODO
        self.linear_plant = linear_plant
        self.tanh_nonlinearity = False

        # initial state
        self.register_buffer('x_target', x_target.reshape(1, -1))  # shape = (1, state_dim)
        x_init = x_init.reshape(1, -1)   # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.x_target.shape[1]/2)) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)

        # check dimensions
        self.n_agents = int(self.x_target.shape[1]/4)
        self.state_dim = 4*self.n_agents
        self.in_dim = 2*self.n_agents
        self.output_dim = int(self.state_dim/2) #TODO
        assert self.x_target.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.x = None

        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.linear_plant else 0.1
        m = self.mass
        B = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [1/m, 0],
                                     [0, 1/m]])
                       ) * self.h
        self.register_buffer('B', B)

        _A1 = torch.eye(4*self.n_agents)
        _A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                torch.eye(2)
                                ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-self.k/self.mass, -self.k/self.mass])),
                                   torch.diag(torch.tensor([-self.b/self.mass, -self.b/self.mass]))
                                   ), dim=1),
                        ), dim=0)
        _A2 = torch.kron(torch.eye(self.n_agents), _A2)
        A_lin = _A1 + self.h * _A2
        self.register_buffer('A_lin', A_lin)

        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)
        self.register_buffer('mask', mask)

        C = torch.kron(torch.eye(self.n_agents),
                            torch.tensor([[1., 0, 0, 0],
                                          [0, 1., 0, 0]])
                            )
        self.register_buffer('C', C)

        D = torch.zeros(self.output_dim, self.output_dim)
        self.register_buffer('D', D)

        self.y_init = F.linear(self.x_init, self.C) + F.linear(self.u_init, self.D)

    def A_nonlin(self):
        assert not self.linear_plant
        A3 = torch.norm(
            self.x.view(-1, 2 * self.n_agents, 2) * self.mask, dim=-1, keepdim=True
        )           # shape = (batch_size, 2 * n_agents, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1, device=A3.device)
        )           # shape = (batch_size, 4 * n_agents, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        )           # shape = (batch_size, 4 * n_agents, 4 * n_agents)
        A = self.A_lin + self.h * A3
        return A    # shape = (batch_size, 4 * n_agents, 4 * n_agents)

    def noiseless_forward(self, t, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        self.x = self.x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)
        if self.linear_plant:
            # x is batched but A is not => can use F.linear to compute xA^T
            x_next = F.linear(self.x - self.x_target, self.A_lin) + F.linear(u, self.B) + self.x_target
        else:
            if not self.tanh_nonlinearity:
                # A depends on x, hence is batched. perform batched matrix multiplication
                x_next = torch.bmm(self.x - self.x_target, self.A_nonlin(x).transpose(1,2)) + F.linear(u, self.B) + self.x_target
            else:
                x_next = (F.linear(self.x - self.x_target, self.A_lin)
                     + self.h * self.b2 / self.mass * self.mask.view(-1) * torch.tanh(x - self.x_target)
                     + F.linear(u, self.B) + self.x_target)
        y_next = F.linear(x_next, self.C) + F.linear(u, self.D)
        return y_next    # shape = (batch_size, 1, output_dim)

    def forward(self, t, u, w):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """
        return self.noiseless_forward(t, u) + w.view(-1, 1, self.output_dim)

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise

        Args:
            - data: sequence of disturbance samples of shape
                (batch_size, T, state_dim).

        Rerurn:
            - x_log of shape = (batch_size, T, state_dim)
            - u_log of shape = (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        self.x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1) #+noise on initial conditions?
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1) #apply pb also at t=0 on initial conditions mismatch?

        # Simulate
        for t in range(data.shape[1]):
            y = self.forward(t=t, u=u, w=data[:, t:t+1, :])
            u = controller(y)                                       # shape = (batch_size, 1, in_dim)

            if t == 0:
                y_log, u_log = y, u
            else:
                y_log = torch.cat((y_log, y), 1)
                u_log = torch.cat((u_log, u), 1)

        controller.reset()
        if not train:
            y_log, u_log = y_log.detach(), u_log.detach()

        return y_log, None, u_log



    # simulation
    def openloop_rollout(self, u, noise=None, train=False):
        """
        rollout REN for rollouts of the process noise

        Args:
            - u: sequence of inputs of shape
                (batch_size, T, input_dim).
            - noise: sequence of disturbance samples of shape
                (batch_size, T, state_dim).

        Rerurn:
            - x_log of shape = (batch_size, T, state_dim)
        """
        if noise is None:
            noise = torch.zeros(u.shape[0], u.shape[1], self.output_dim)
        # init
        assert noise.shape[0]==u.shape[0] and noise.shape[1]==u.shape[1]
        noise = noise.detach().clone()
        u = u.detach().clone()
        y = self.y_init.detach().clone().repeat(noise.shape[0], 1, 1) + noise[:, 0:1, :]
        y_log = y
        # Simulate
        for t in range(noise.shape[1]-1):
            y = self.forward(t=t, u=u[:, t:t+1, :], w=noise[:, t+1:t+2, :])    # shape = (batch_size, 1, state_dim)                                     # shape = (batch_size, 1, in_dim)
            y_log = torch.cat((y_log, y), 1)

        if not train:
            y_log = y_log.detach()

        return y_log

    def generate_input_noise(self, input_noise_std, horizon, num_samples=1, noise_only_on_init = True):
        if noise_only_on_init:
            noise = torch.zeros(num_samples, horizon, self.input_dim)
            noise[:,0,:] = input_noise_std * torch.randn(num_samples, self.input_dim).reshape(num_samples, 1, self.input_dim)
        else:
            noise = input_noise_std * torch.randn(num_samples, horizon, self.input_dim)
        return noise

    def generate_output_noise(self, output_noise_std, horizon, num_samples=1, noise_only_on_init = True):
        if noise_only_on_init:
            noise = torch.zeros(num_samples, horizon, self.output_dim)
            noise[:,0,:] = output_noise_std * torch.randn(num_samples, self.output_dim).reshape(num_samples, 1, self.output_dim)
        else:
            noise = output_noise_std * torch.randn(num_samples, horizon, self.output_dim)
        return noise