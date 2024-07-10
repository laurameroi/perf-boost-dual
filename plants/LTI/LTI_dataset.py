import torch
import numpy as np # TODO: move to torch
from plants import CostumDataset


class LTIDataset(CostumDataset):
    def __init__(self, random_seed, horizon, disturbance, state_dim):
        # experiment and file names
        exp_name = 'LTI'
        file_name = disturbance['type'].replace(" ", "_")+'_data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.disturbance = disturbance
        self.state_dim = state_dim
        self.random_state = np.random.RandomState(random_seed)  # TODO: remove and use torch

    # ---- data generation ----
    def _generate_data(self, num_samples):
        if self.disturbance['type'] in ['N 0-mean', 'N biased']:
            data = self.random_state.multivariate_normal(
                mean=self.disturbance['mean'], cov=self.disturbance['cov'],
                size=(num_samples, self.horizon)
            )
        elif self.disturbance['type'] == 'N multi-modal':
            # A stream of indices from which to choose the component
            mixture_idx = self.random_state.choice(
                len(self.disturbance['weight']), size=num_samples*self.horizon,
                replace=True, p=self.disturbance['weight']
            )
            # y is the mixture sample
            data = np.array([
                    [self.random_state.multivariate_normal(
                        self.disturbance['mean'][mixture_idx[s_ind*num_samples + t_ind]], self.disturbance['cov']
                    ) for t_ind in range(self.horizon)]
                for s_ind in range(num_samples)]
            )
        elif self.disturbance['type'] == 'Uniform':
            data = self.random_state.uniform(
                low=self.disturbance['low'], high=self.disturbance['high'], size=(num_samples, self.horizon)
            )
        else:
            raise NotImplementedError
        data = np.reshape(
            data,
            (num_samples, self.horizon, self.state_dim)
        )
        data = torch.from_numpy(data)
        return data



