import torch
from plants import CostumDataset


class RobotsDataset(CostumDataset):
    def __init__(self, random_seed, horizon, std_ini=0.2, n_agents=2):
        # experiment and file names
        exp_name = 'minimal_example'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents

        # initial state TODO: set as arg
        self.x0 = torch.tensor([2., -2, 0, 0,
                                -2, -2, 0, 0,
                                ])
        self.xbar = torch.tensor([-2, 2, 0, 0,
                                  2., 2, 0, 0,
                                  ])

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 4*self.n_agents
        data = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            data[rollout_num, 0, :] = \
                (self.x0 - self.xbar) + self.std_ini * torch.randn(self.x0.shape)

        assert data.shape[0]==num_samples
        return data