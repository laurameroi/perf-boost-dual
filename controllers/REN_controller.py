import torch, copy
import torch.nn as nn
from config import device

from .acyclic_ren import AcyclicREN

class RENController(nn.Module):
    """
    TODO: add description
    TODO: change class name?
    state-feedback controller with stability guarantees.
    NOTE: controller has input "u", output "y", and internal state "x". When used in closed-loop,
        the controller input "u" would be the measured state and the controller output ("y")
        would be the input to the plant. The internal state of the controller should not be mistaken
        with the internal state of the plant.
    """
    def __init__(
        self, noiseless_forward, input_init, output_init,
        # acyclic REN properties
        dim_internal: int, l: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        # misc
        output_amplification: float=20,
    ):
        """
         Args:
            noiseless_forward: system dynamics without process noise. can be TV.
            input_init: initial input to the controller.
            output_init: initial output from the controller before anything is calculated.
            output_amplification (float): TODO
            * the following are the same as AcyclicREN args:
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            l (int): Complexity of the implicit layer.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        self.output_amplification = output_amplification

        # set initial conditions
        self.input_init = input_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        # set dimensions
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]

        # define the REN
        self.psi_u = AcyclicREN(
            dim_in=self.dim_in, dim_out=self.dim_out, dim_internal=dim_internal,
            l=l, initialization_std=initialization_std,
            internal_state_init=None,   # initialize at 0
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb
        )

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward


        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = copy.deepcopy(self.input_init)
        self.last_output = copy.deepcopy(self.output_init)
        self.psi_u.x = torch.zeros(1, self.psi_u.dim_internal, device=device)   # set the internal state to 0


    def forward(self, u_in: torch.Tensor):
        """
        Forward pass of the controller.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            NOTE: when used in closed-loop, "u_in" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # batch u_in to (batch_size, dim_in, 1)
        if len(u_in.shape)<=2:
            u_in = u_in.reshape(1, 1, -1)
        else:
            assert len(u_in.shape)==3

        # apply noiseless forward to get noise less input (noise less state of the plant)
        u_noiseless = self.noiseless_forward(
            t=self.t,
            x=self.last_input,  # last input to the controller is the last state of the plant
            u=self.last_output  # last output of the controller is the last input to the plant
        ) # shape = (self.batch_size, 1, self.dim_in)

        # reconstruct the noise
        w_ = u_in - u_noiseless # shape = (self.batch_size, 1, self.dim_in)

        # apply REN
        output = self.psi_u.forward(w_)
        output = output*self.output_amplification   # shape = (self.batch_size, 1, self.dim_out)

        # assert xi_.shape==(self.batch_size, 1, self.psi_u.dim_internal), xi_.shape
        # update internal states
        self.last_input, self.last_output = u_in, output
        self.t += 1
        return output

    # setters and getters
    def get_parameter_shapes(self):
        return self.psi_u.get_parameter_shapes()

    def get_named_parameters(self):
        return self.psi_u.get_named_parameters()

    def get_parameters_as_vector(self):
        return torch.cat(list(self.named_parameters().values()), dim=-1)

    def set_parameter(self, name, value):
        current_val = getattr(self.psi_u, name)
        value = torch.nn.Parameter(value.reshape(current_val.shape))
        setattr(self.psi_u, name, value)
        self.psi_u.set_model_param()    # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        # value is reshaped to the parameter shape
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            # select indx
            if value.ndim == 1:
                value_tmp = value[idx:idx_next]
            elif value.ndim == 2:
                value_tmp = value[:, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():
                self.set_parameter(name, value_tmp)
            idx = idx_next
        assert idx_next == value.shape[-1]


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
