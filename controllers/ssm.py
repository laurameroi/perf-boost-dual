import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from controllers.non_linearities import MLP, HamiltonianSIE, CouplingLayer


class LRU(nn.Module):
    """
    Implements a Linear Recurrent Unit (LRU) following the parametrization of "Resurrecting Linear Recurrences" paper.
    The LRU is simulated using Parallel Scan (fast!) when scan=True (default), otherwise recursively (slow)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 state_features: int,
                 scan: bool = True,  # This has been removed
                 rmin: float = 0.9,
                 rmax: float = 1.,
                 max_phase: float = 6.283,
                 internal_state_init=None
                 ):
        super().__init__()

        # set dimensions
        self.dim_internal = state_features
        self.input_dim = in_features
        self.scan = scan
        self.output_dim = out_features

        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(lambda_mod) - torch.square(lambda_mod))))
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        # self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

        # define trainable params
        self.training_param_names = ['D', 'nu_log', 'theta_log', 'gamma_log', 'B', 'C']

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.complex(torch.zeros(state_features), torch.zeros(state_features))
            # torch.zeros(1, 1, self.dim_internal)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            assert internal_state_init.dim == 1 and internal_state_init.shape[0] == 4
            if internal_state_init.is_complex():
                self.x = self.dim_internal
            else:
                self.x = torch.complex(internal_state_init, torch.zeros(self.dim_internal))
        self.register_buffer('init_x', self.x.detach().clone())

    def forward(self, u_in):
        """
        Forward pass of SSM.
        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.input_dim).
        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.output_dim).
        """
        batch_size = u_in.shape[0]

        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.theta_log))
        lambda_c = torch.complex(lambda_re, lambda_im)  # A matrix
        gammas = torch.exp(self.gamma_log)

        self.x = lambda_c * self.x + gammas * F.linear(torch.complex(u_in, torch.zeros(1)), self.B)
        y_out = 2 * F.linear(self.x, self.C).real + F.linear(u_in, self.D)
        return y_out


# Class for implementing LRU + a user-defined scaffolding, this is our SSM block.
class SSM(nn.Module):
    # Scaffolding can be modified. In this case we have LRU, MLP plus linear skip connection.
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dim_internal: int,
                 scan: bool = False,
                 dim_hidden: int = 30,
                 rmin: float = 0.95,
                 rmax: float = 0.99,
                 max_phase: float = 6.283,
                 internal_state_init=None,
                 non_linearity: str = "MLP"
                 ):
        super().__init__()

        # set dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_internal = dim_internal
        self.dim_hidden = dim_hidden

        if non_linearity == "MLP":
            self.scaffold = MLP(output_dim, dim_hidden, output_dim)
        elif non_linearity == "coupling_layers":
            # Option 2: coupling (or invertible) layers
            self.scaffold = CouplingLayer(output_dim, dim_hidden)
        elif non_linearity == "hamiltonian":
            # Option 3: Hamiltonian net
            self.scaffold = HamiltonianSIE(n_layers=4, nf=output_dim, bias=False)
        elif non_linearity == "tanh":
            self.scaffold = torch.tanh
        else:
            # End options
            raise NotImplementedError("The non_linearity %s is not implemented" % non_linearity)
        self.lru = LRU(input_dim, output_dim, dim_internal, scan, rmin, rmax, max_phase, internal_state_init)
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.zeros_(self.lin.weight.data)

    def forward(self, u):
        result = self.scaffold(self.lru(u)) + self.lin(u)
        return result

    def get_named_parameters(self):
        print("CLARA: Not working! self.training_param_names is not defined!")
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

    def get_parameter_shapes(self):
        print("CLARA: Not working! self.training_param_names is not defined!")
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict


# Class implementing a cascade of N SSMs. Linear pre- and post-processing can be modified
class DeepSSM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 dim_internal: int,
                 dim_middle: int,
                 dim_hidden: int = 30,
                 # scan: bool,
                 # n_ssm: int,
                 rmin: float = 0.9,
                 rmax: float = 1,
                 max_phase: float = 6.283,
                 internal_state_init=None,
                 non_linearity="MLP"
                 ):
        super().__init__()

        # set dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_internal = dim_internal
        self.dim_hidden = dim_hidden

        self.ssm1 = SSM(input_dim, dim_middle, dim_internal, dim_hidden=dim_hidden, non_linearity=non_linearity)
        self.ssm2 = SSM(dim_middle, output_dim, dim_internal, dim_hidden=dim_hidden, non_linearity=non_linearity)

    def forward(self, u_in):
        y_out = self.ssm2(self.ssm1(u_in))
        return y_out

    def reset(self):
        self.ssm1.lru.x = self.ssm1.lru.init_x  # reset the SSM.LRU state to the initial value
        self.ssm2.lru.x = self.ssm2.lru.init_x  # reset the SSM.LRU state to the initial value

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict


if __name__ == "__main__":
    input_dim = 2
    output_dim = 2
    dim_internal = 4
    dim_hidden = 8
    batch_size = 3
    ssm = SSM(input_dim, output_dim, dim_internal, scan=False, dim_hidden=dim_hidden, non_linearity="hamiltonian")
    deep_ssm = DeepSSM(input_dim, output_dim, dim_internal, dim_middle=6, dim_hidden=dim_hidden, non_linearity="hamiltonian")

    # Print dimensions:
    print("B has dimensions: ", ssm.lru.B.shape)
    print("C has dimensions: ", ssm.lru.C.shape)
    print("D has dimensions: ", ssm.lru.D.shape)
    print("nu_log has dimensions: ", ssm.lru.nu_log.shape)
    print("theta_log has dimensions: ", ssm.lru.theta_log.shape)
    print("gamma_log has dimensions: ", ssm.lru.gamma_log.shape)
    print("the state has dimensions: ", ssm.lru.x.shape)

    # # Test methods
    # param_dict = ssm.get_named_parameters()
    # print(param_dict)

    t = torch.linspace(0, 1, 100)
    u = torch.zeros(batch_size, t.shape[0], input_dim)
    for i in range(batch_size):
        u[i, 0, :] = torch.randn(input_dim)
    y_ssm = torch.zeros(batch_size, t.shape[0], output_dim)
    y_deep_ssm = torch.zeros(batch_size, t.shape[0], output_dim)
    x_ssm = torch.complex(torch.zeros(batch_size, t.shape[0], dim_internal), torch.zeros(1, t.shape[0], dim_internal))
    x_deep_ssm = torch.complex(torch.zeros(batch_size, t.shape[0], dim_internal),
                               torch.zeros(1, t.shape[0], dim_internal))
    for i in range(t.shape[0]):
        y_ssm[:, i:i + 1, :] = ssm(u[:, i:i + 1, :])
        x_ssm[:, i:i + 1, :] = ssm.lru.x
        y_deep_ssm[:, i:i + 1, :] = deep_ssm(u[:, i:i + 1, :])
        x_deep_ssm[:, i:i + 1, :] = deep_ssm.ssm1.lru.x

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t, y_ssm[0, :, :].detach())
    plt.title("Output SSM")
    plt.figure()
    plt.plot(t, u[0, :, :].detach())
    plt.title("Input")
    plt.figure()
    plt.plot(t, x_ssm[0, :, :].real.detach())
    plt.title("State SSM")
    plt.figure()
    plt.plot(t, y_deep_ssm[0, :, :].detach())
    plt.title("Output Deep SSM")
    plt.figure()
    plt.plot(t, x_deep_ssm[0, :, :].real.detach())
    plt.title("State 1st SSM of Deep SSM")
    plt.show()
