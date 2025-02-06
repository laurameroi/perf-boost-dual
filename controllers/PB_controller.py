import torch, time, copy
import torch.nn as nn
import numpy as np

from config import device
from .contractive_ren import ContractiveREN
from controllers.ssm import DeepSSM
from assistive_functions import to_tensor


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state of the plant
          and the controller output is the input to the plant.
    This controller has a memory for the last input ("self.last_input") and the last output ("self.last_output").
    """

    def __init__(self,
                 internal_model,
                 input_init: torch.Tensor,
                 output_init: torch.Tensor,
                 nn_type: str = "REN",
                 non_linearity: str = None,
                 # acyclic REN properties
                 dim_internal: int = 8,
                 dim_nl: int = 8,
                 initialization_std: float = 0.5,
                 pos_def_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0,
                 ren_internal_state_init=None,
                 # misc
                 output_amplification: float = 20,
                 ):
        """
         Args:
            noiseless_forward:            System dynamics without process noise. It can be TV.
            input_init (torch.Tensor):    Initial input to the controller.
            output_init (torch.Tensor):   Initial output from the controller before anything is calculated.
            nn_type (str):                Which NN model to use for the Emme operator (Options: 'REN' or 'SSM')
            non_linearity (str):          Non-linearity used in SSMs for scaffolding.
            output_amplification (float): TODO
            ##### the following are the same as AcyclicREN args:
            dim_internal (int):           Internal state (x) dimension.
            dim_nl (int):                 Dimension of the input ("v") and output ("w") of the NL static block of REN.
            initialization_std (float):   [Optional] Weight initialization. Set to 0.1 by default.
            pos_def_tol (float):          [Optional] Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float):  [Optional] Lower bound on the contraction rate. Default to 1.
            ren_internal_state_init (torch.Tensor): [Optional] Initial state of the REN. Default to 0 when None.
        """
        super().__init__()

        self.output_amplification = output_amplification

        # set initial conditions
        self.input_init = input_init.reshape(1, -1)
        self.output_init = output_init.reshape(1, -1)

        # set dimensions
        self.input_dim = self.input_init.shape[-1]
        self.output_dim = self.output_init.shape[-1]

        # define the system dynamics without process noise
        self.internal_model = internal_model
        # freeze parameters of the internal model
        for p in self.internal_model.parameters():
            p.requires_grad = False


        # set type of nn for emme
        self.nn_type = nn_type
        # define Emme as REN or SSM
        if nn_type == "REN":
            self.emme = ContractiveREN(
                input_dim=self.input_dim, output_dim=self.output_dim, dim_internal=dim_internal,
                dim_nl=dim_nl, initialization_std=initialization_std,
                internal_state_init=ren_internal_state_init,
                pos_def_tol=pos_def_tol, contraction_rate_lb=contraction_rate_lb
            ).to(device)
        elif nn_type == "SSM":
            # define the SSM
            self.emme = DeepSSM(self.input_dim,
                                self.output_dim,
                                dim_internal,
                                dim_middle=6,
                                dim_hidden=dim_nl,
                                non_linearity=non_linearity
                                ).to(device)
        else:
            raise ValueError("Model for emme not implemented")

        # Internal variables
        self.t = None
        self.last_input = None
        self.last_output = None
        # Initialize internal variables
        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.emme.reset()  # reset emme states to the initial value
        self.internal_model.reset()

    def forward(self, input_t: torch.Tensor):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.input_dim).
            # NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.output_dim).
        """
        # apply noiseless forward to get noise less input (noise less state of the plant)
        # with torch.no_grad():
        u_noiseless = self.internal_model.forward(
            u=self.last_output  # last output of the controller is the last input to the plant
        )  # shape = (self.batch_size, 1, self.input_dim)

        # reconstruct the noise
        w_ = input_t - u_noiseless  # shape = (self.batch_size, 1, self.input_dim)
        # apply REN or SSM
        output = self.emme.forward(w_)
        output = output * self.output_amplification  # shape = (self.batch_size, 1, self.output_dim)
        # update internal states
        self.last_input, self.last_output = input_t, output
        self.t += 1
        return output

    # setters and getters
    def get_parameter_shapes(self):
        if self.nn_type == 'SSM':
            raise ValueError("not implemented")
        return self.emme.get_parameter_shapes()

    def get_named_parameters(self):
        if self.nn_type == 'SSM':
            raise ValueError("not implemented")
        return self.emme.get_named_parameters()

    def get_parameters_as_vector(self):
        # TODO: implement without numpy
        return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.emme.parameters()])

    def set_parameter(self, name, value):
        if self.nn_type == 'SSM':
            print("This function might not work for SSMs.....")
        current_val = getattr(self.emme, name)
        value = torch.nn.Parameter(torch.tensor(value.reshape(current_val.shape)))
        setattr(self.emme, name, value)
        if self.nn_type == 'REN':
            self.emme._update_model_param()  # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        # flatten vec if not batched
        if value.nelement()==self.num_params:
            value = value.flatten()

        if self.nn_type == 'SSM':
            print("This function might not work for SSMs.....")
        idx = 0
        idx_next = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape
            elif len(shape) == 2:
                dim = shape[0] * shape[1]
            else:
                dim = shape[-1]*shape[-2]
            idx_next = idx + dim
            # select index
            if len(value.shape) == 1:
                value_tmp = value[idx:idx_next]
            elif len(value.shape) == 2:
                value_tmp = value[:, idx:idx_next]
            elif value.ndim == 3:
                value_tmp = value[:, :, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():
                self.set_parameter(name, value_tmp.reshape(shape))
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, *args, **kwargs):  # CLARA: Why do we need this function? Isn't it implemented in nn.Module?
        return self.forward(*args, **kwargs)

    def fit(
        self, sys, train_dataloader, valid_data, lr, loss_fn, epochs, 
        log_epoch, logger, return_best, early_stopping, n_logs_no_change=None, tol_percentage=None
    ):
        logger.info('\n------------ Begin training ------------')

        # Set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Initialize best valid loss and best parameters
        if return_best:
            best_valid_loss = float('inf')
            best_params = self.state_dict()
        # Queue to store the last n_logs_no_change validation improvements
        if early_stopping:
            assert not (n_logs_no_change is None or tol_percentage is None), 'Early stopping requires n_logs_no_change and tol_percentage'  
            valid_imp_queue = [100]*n_logs_no_change   # Set to 100 to avoid stopping at the beginning

        # Record start time
        start_time = time.time()

        fitted = False
        try:
            # Train the controller
            for epoch in range(epochs + 1):
                train_loss_batch = 0    # Accumulate training loss over the batch
                for train_data_batch in train_dataloader:
                    optimizer.zero_grad()
                    # Simulate over horizon steps
                    y_log, _, u_log = sys.rollout(controller=self, output_noise_data=train_data_batch, train=True)
                    # Calculate loss of all rollouts in the batch
                    loss = loss_fn(y_log, u_log)
                    train_loss_batch += loss.item()
                    # Backpropagation and optimization step
                    loss.backward()
                    optimizer.step()

                # Log training information
                if epoch % log_epoch == 0:
                    msg = f'Epoch: {epoch} --- train loss: {loss.item():.2f}'
                    if return_best or early_stopping:
                        # Rollout the current controller on the validation data
                        with torch.no_grad():
                            y_log_valid, _, u_log_valid = sys.rollout(controller=self, output_noise_data=valid_data, train=False)
                            # Calculate validation loss
                            loss_valid = loss_fn(y_log_valid, u_log_valid)
                        msg += f' ---||--- validation loss: {loss_valid.item():.2f}'
                        # Compare with the best validation loss
                        imp = 100 * (best_valid_loss-loss_valid.item())/best_valid_loss # Valid loss improvement
                        if imp>0:
                            best_valid_loss = loss_valid.item()
                            if return_best:
                                best_params = copy.deepcopy(self.state_dict())  # Record state dict if best on valid
                                msg += ' (best so far)'
                        # Early stopping
                        if early_stopping:
                            # Add the current improvement to the queue
                            valid_imp_queue.pop(0)
                            valid_imp_queue.append(imp)
                            # Check if there is no improvement
                            if all([valid_imp_queue[i] <tol_percentage for i in range(n_logs_no_change)]):
                                msg += ' ---||--- early stopping'
                                fitted = True
                    elapsed_time = time.time() - start_time
                    msg += f' ---||--- elapsed time: {elapsed_time:.0f} s'
                    logger.info(msg)
                    if fitted:
                        break
        except Exception as e:
            logger.error(f'An error occurred during training: {e}')
            raise e

        # Set to best seen parameters during training
        if return_best:
            self.load_state_dict(best_params)


    def rollout(self, sys, output_noise_data, input_noise_data=None, ref=None, train=False):
        """
        rollout PB for rollouts of the process noise

        Args:
            - data: sequence of disturbance samples of shape
                (batch_size, T, output_dim).

        Rerurn:
            - y_log of shape = (batch_size, T, output_dim)
            - u_log of shape = (batch_size, T, input_dim)
        """
        if ref is None:
            ref = torch.zeros(*output_noise_data.shape[0:2], sys.input_dim, device=sys.x_init.device)
        if input_noise_data is None:
            input_noise_data = torch.zeros(*output_noise_data.shape[0:2], sys.input_dim, device=sys.x_init.device)
        # init
        self.reset()
        sys.x = sys.x_init.detach().clone().repeat(output_noise_data.shape[0], 1, 1) #+noise on initial conditions?
        u = sys.u_init.detach().clone().repeat(output_noise_data.shape[0], 1, 1) #apply pb also at t=0 on initial conditions mismatch?
        u += ref[:, 0:1, :] + input_noise_data[:, 0:1, :]
        
        # Simulate
        for t in range(output_noise_data.shape[1]):
            y = sys.forward(t=t, u=u, output_noise=output_noise_data[:, t:t+1, :]) # y_t, x_{t+1} gets stored in sys.x
            if t == 0:
                y_log, u_log = y, u
            else:
                y_log = torch.cat((y_log, y), 1)
                u_log = torch.cat((u_log, u), 1)

            # compute u for the next time step
            if not t == output_noise_data.shape[1]-1:
                u = self(y)+ref[:, t+1:t+2, :] + input_noise_data[:, t+1:t+2, :] # u_{t+1} shape = (batch_size, 1, input_dim)

        self.reset()
        if not train:
            y_log, u_log = y_log.detach(), u_log.detach()

        return y_log, None, u_log