import torch
from plants import CostumDataset
import numpy as np


class RobotsDataset(CostumDataset):
    def __init__(self, random_seed, horizon, n_agents=1):
        # experiment and file names
        exp_name = 'minimal_example'
        file_name = 'data_T'+str(horizon)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

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
    
    #CLOSED LOOP data generation of dimension num_samples
    def generate_openloop_dataset(self, sys, noise_only_on_init, num_samples=50, ts=0.05):
        device = sys.x_init.device
        
        # generate noise on plant input and output
        input_noise = sys.generate_input_noise(num_samples=num_samples, horizon=self.horizon, noise_only_on_init=noise_only_on_init)
        output_noise = sys.generate_output_noise(num_samples=num_samples, horizon=self.horizon, noise_only_on_init=noise_only_on_init)
        input_noise, output_noise = input_noise.to(device), output_noise.to(device)

        # PE reference used as input to sys, each is a 2D sinusoidal signal
        ref_data = generate_sinusoidal_dataset(num_samples=num_samples, ts=ts, horizon=self.horizon, input_dim=sys.input_dim)
        ref_data = ref_data.to(device)
        assert ref_data.shape==input_noise.shape
        ref_data_noisy = ref_data + input_noise

        output_data_noisy = sys.openloop_rollout(ref_data_noisy, output_noise=output_noise)
        assert output_data_noisy.shape == output_noise.shape

        return output_data_noisy, ref_data_noisy

    #CLOSED LOOP data generation of dimension num_samples
    def generate_closedloop_dataset(self, sys, controller, noise_only_on_init, num_samples=50, ts=0.05):
        device = sys.x_init.device
        # generate noise
        data = self._generate_data(num_samples, noise_only_on_init=noise_only_on_init)
        x_log,_, u_log =sys.rollout(controller, data)

        x_log, u_log = x_log.to(device), u_log.to(device)

        return x_log, u_log

# Define a function to generate the sinusoidal signals for horizontal and vertical forces
def generate_sinusoidal(frequency, amplitude, phase, time):
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

def generate_sinusoidal_dataset(num_samples=50, ts=0.05, horizon=10, input_dim=2):
    time = np.arange(0, horizon*ts, ts)  # Time from 0 to 15 seconds with a sampling time of 0.05 seconds
    data = torch.zeros(num_samples, len(time), input_dim)
    for n in range(num_samples):
        # Generate random parameters for the current sample
        frequency_x = np.random.uniform(1, 2)  # Horizontal frequency between 0.5 Hz and 5 Hz
        frequency_y = np.random.uniform(1, 2)  # Vertical frequency between 0.5 Hz and 5 Hz
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