import torch
from plants import CostumDataset
import numpy as np


class RobotsDataset(CostumDataset):
    def __init__(self, random_seed, horizon, std_noise=0.2, n_agents=2):
        # experiment and file names
        exp_name = 'minimal_example'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(std_noise)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_noise = std_noise
        self.n_agents = n_agents

        if n_agents==2:
            self.x_init = torch.tensor([2., -2, 0, 0,
                                    -2, -2, 0, 0,
                                    ])
            self.x_target = torch.tensor([-2, 2, 0, 0,
                                    2., 2, 0, 0,
                                    ])
        elif n_agents==1:
            self.x_init = torch.tensor([2., -2, 0, 0])
            self.x_target = torch.tensor([0, 0, 0, 0])  #TODO
        else:
            raise ValueError('n_agents must be 1 or 2')

    # ---- data generation ----
    def _generate_noise(self, num_samples, noise_only_on_init=False):
        state_dim = 4*self.n_agents
        self.noise = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            if noise_only_on_init:
                self.noise[rollout_num, 0, :] = self.std_noise * torch.randn(self.x_init.shape)
            else:
                self.noise[rollout_num, :, :] = self.std_noise * torch.randn((self.horizon, *self.x_init.shape))

        assert self.noise.shape[0]==num_samples
        # self.noise = torch.zeros(num_samples, self.horizon, state_dim)    # use for noise-free debug
        return self.noise
    
    #CLOSED LOOP data generation of dimension num_signals
    def generate_openloop_dataset(self, sys, noise_only_on_init, num_signals=50, ts=0.05):
        device = sys.x_init.device
        # generate noise
        self._generate_noise(num_signals, noise_only_on_init=noise_only_on_init)

        # PE input to sys, each is a 2D sinusoidal signal
        input_data = generate_input_dataset(num_signals=num_signals, ts=ts, horizon=self.horizon, input_dim=sys.in_dim)

        #hard coded output dimension
        output_data = torch.zeros(num_signals, self.horizon, sys.state_dim, device=device)

        self.noise, input_data = self.noise.to(device), input_data.to(device)
        for n in range(num_signals):
            output_data[n, 0, :] = (self.x_init).to(device) + self.noise[n, 0, :]  # nominal initial state
            for t in range(output_data.shape[1]-1):
                u = input_data[n, t, :]
                # u=torch.zeros(2, device=device)   # use for debugging
                # Compute next state using the forward dynamics
                output_data[n, t+1, :] = sys.forward(t, output_data[n, t, :], u, self.noise[n, t+1, :])
        return output_data, input_data

# Define a function to generate the sinusoidal signals for horizontal and vertical forces
def generate_sinusoidal(frequency, amplitude, phase, time):
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

def generate_input_dataset(num_signals=50, ts=0.05, horizon=100, input_dim=2):
    time = np.arange(0, horizon*ts, ts)  # Time from 0 to 15 seconds with a sampling time of 0.05 seconds
    data = torch.zeros(num_signals, len(time), input_dim)
    for n in range(num_signals):
        # Generate random parameters for the current sample
        frequency_x = np.random.uniform(0.5, 1)  # Horizontal frequency between 0.5 Hz and 5 Hz
        frequency_y = np.random.uniform(0.5, 1)  # Vertical frequency between 0.5 Hz and 5 Hz
        amplitude_x = np.random.uniform(0.5, 3)  # Horizontal amplitude between 0.5 and 3
        amplitude_y = np.random.uniform(0.5, 3)  # Vertical amplitude between 0.5 and 3
        phase_x = np.random.uniform(-np.pi, np.pi)  # Horizontal phase between -π and π
        phase_y = np.random.uniform(-np.pi, np.pi)  # Vertical phase between -π and π

        # Generate the signals using the random parameters
        horizontal_force = generate_sinusoidal(frequency_x, amplitude_x, phase_x, time)
        data[n, :, 0] = torch.from_numpy(horizontal_force)
        vertical_force = generate_sinusoidal(frequency_y, amplitude_y, phase_y, time)
        data[n, :, 1] = torch.from_numpy(vertical_force)
    return data