import torch
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, input_noise_std, output_noise_std, y_target: torch.Tensor, linear_plant: bool, x_init, u_init=None, k: float=1.0):
        """

        Args:
            y_target: concatenated target point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to y_target when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        torch.manual_seed(0) #TODO
        self.input_noise_std = input_noise_std
        self.output_noise_std = output_noise_std
        self.linear_plant = linear_plant
        self.tanh_nonlinearity = False

        # check dimensions
        self.register_buffer('y_target', y_target.reshape(1, -1))  # shape = (1, output_dim)
        self.register_buffer(
            'x_target', 
            torch.tensor([self.y_target[0,0], self.y_target[0,1], 0, 0]).reshape(1, -1)
        )  # shape = (1, state_dim)
        self.n_agents = int(self.y_target.shape[1]/2)
        self.state_dim = 4*self.n_agents
        self.input_dim = 2*self.n_agents
        self.output_dim = 2*self.n_agents

        # initial state
        x_init = x_init.reshape(1, self.state_dim)   # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, self.input_dim) if u_init is None else u_init.reshape(1, self.input_dim)   # shape = (1, input_dim)
        self.register_buffer('u_init', u_init)

        self.register_buffer('x', self.x_init.detach().clone())

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

        self.register_buffer('y_init_nominal', F.linear(self.x_init, self.C))
        
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

    def reset(self):
        self.x = self.x_init.detach().clone().reshape(1, 1, self.state_dim)

    def noiseless_forward(self, t, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, input_dim)

        Returns:
            output at time t of the noise-free dynamics.
        """
        self.x = self.x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.input_dim)
        y = F.linear(self.x, self.C) + F.linear(u, self.D)

        # compute next state
        if self.linear_plant:
            # x is batched but A is not => can use F.linear to compute xA^T
            x_next = F.linear(self.x - self.x_target, self.A_lin) + F.linear(u, self.B) + self.x_target
        else:
            if not self.tanh_nonlinearity:
                # A depends on x, hence is batched. perform batched matrix multiplication
                x_next = torch.bmm(self.x - self.x_target, self.A_nonlin(self.x).transpose(1,2)) + F.linear(u, self.B) + self.x_target
            else:
                x_next = (F.linear(self.x - self.x_target, self.A_lin)
                     + self.h * self.b2 / self.mass * self.mask.view(-1) * torch.tanh(self.x - self.x_target)
                     + F.linear(u, self.B) + self.x_target)
        self.x = x_next
        return y    # shape = (batch_size, 1, output_dim)

    def forward(self, u, t=None, output_noise=None):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, input_dim)
            - w (torch.Tensor): output noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            noisy output of the plant at time t
        """
        if len(u.shape)==2:
            u = u.reshape(1, *u.shape)
        if output_noise is None: 
            output_noise = torch.zeros(*u.shape[0:2], self.output_dim, device=u.device)
        return self.noiseless_forward(t, u) + output_noise.view(-1, 1, self.output_dim)

    # simulation
    def rollout(self, controller, output_noise_data, ref=None, train=False):
        return controller.rollout(sys=self, output_noise_data=output_noise_data, ref=ref, train=train)
    
    # def rollout(self, controller, output_noise_data, ref=None, train=False):
    #     """
    #     rollout REN for rollouts of the process noise

    #     Args:
    #         - data: sequence of disturbance samples of shape
    #             (batch_size, T, output_dim).

    #     Rerurn:
    #         - y_log of shape = (batch_size, T, output_dim)
    #         - u_log of shape = (batch_size, T, input_dim)
    #     """
    #     if ref is None:
    #         ref = torch.zeros(*output_noise_data.shape[0:2], self.input_dim, device=self.x_init.device)
    #     # init
    #     controller.reset()
    #     self.x = self.x_init.detach().clone().repeat(output_noise_data.shape[0], 1, 1) #+noise on initial conditions?
    #     u = self.u_init.detach().clone().repeat(output_noise_data.shape[0], 1, 1) #apply pb also at t=0 on initial conditions mismatch?
    #     # print('u', u.shape, 'ref0 ', ref[:, 0:1, :].shape)
    #     u += ref[:, 0:1, :]
        
    #     # Simulate
    #     for t in range(output_noise_data.shape[1]):
    #         y = self.forward(t=t, u=u, output_noise=output_noise_data[:, t:t+1, :]) # y_t, x_{t+1} gets stored in self.x

    #         if t == 0:
    #             y_log, u_log = y, u
    #         else:
    #             y_log = torch.cat((y_log, y), 1)
    #             u_log = torch.cat((u_log, u), 1)

    #         if not t == output_noise_data.shape[1]-1:
    #             u = controller(y)+ref[:, t:t+1, :] # u_{t+1} shape = (batch_size, 1, input_dim)

    #     controller.reset()
    #     if not train:
    #         y_log, u_log = y_log.detach(), u_log.detach()

    #     return y_log, None, u_log



    # simulation
    def openloop_rollout(self, u=None, output_noise=None, train=False):
        """
        rollout REN for rollouts of the process noise

        Args:
            - u: sequence of inputs of shape
                (batch_size, T, input_dim).
            - noise: sequence of disturbance samples of shape
                (batch_size, T, state_dim).openloop_rollout

        Rerurn:
            - y_log of shape = (batch_size, T, output_dim)
        """
        assert not (u is None and output_noise is None), 'at least one of u or output_noise should not be None'
        if u is None: 
            u = torch.zeros(*output_noise.shape[0:2], self.input_dim, device=output_noise.device)
        if output_noise is None:
            output_noise = torch.zeros(*u.shape[0:2], self.output_dim, device=u.device)
        # init
        self.x = self.x_init.detach().clone()
        assert output_noise.shape[0]==u.shape[0] and output_noise.shape[1]==u.shape[1]

        u = u.detach().clone()

        # Simulate
        for t in range(output_noise.shape[1]):
            y = self.forward(t=t, u=u[:, t:t+1, :], output_noise=output_noise[:, t:t+1, :])    # y_t. x_{t+1} gets updated. shape = (batch_size, 1, output_dim)                                     # shape = (batch_size, 1, input_dim)f t == 0:
            if t==0:
                y_log = y
            else:
                y_log = torch.cat((y_log, y), 1)

        if not train:
            y_log = y_log.detach()

        return y_log

    def generate_input_noise(self, horizon, num_samples=1, noise_only_on_init = True):
        if self.input_noise_std>0:
            if noise_only_on_init:
                noise = torch.zeros(num_samples, horizon, self.input_dim)
                noise[:,0,:] = self.input_noise_std * torch.randn(num_samples, self.input_dim).reshape(num_samples, 1, self.input_dim)
            else:
                noise = self.input_noise_std * torch.randn(num_samples, horizon, self.input_dim)
        else:
            noise = torch.zeros(num_samples, horizon, self.input_dim)
        return noise

    def generate_output_noise(self, horizon, num_samples=1, noise_only_on_init = True):
        if self.output_noise_std>0:
            if noise_only_on_init:
                noise = torch.zeros(num_samples, horizon, self.output_dim)
                noise[:,0,:] = self.output_noise_std * torch.randn(num_samples, self.output_dim).reshape(num_samples, 1, self.output_dim)
            else:
                noise = self.output_noise_std * torch.randn(num_samples, horizon, self.output_dim)
        else:
            noise = torch.zeros(num_samples, horizon, self.output_dim)
        return noise