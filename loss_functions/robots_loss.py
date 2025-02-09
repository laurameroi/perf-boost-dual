import torch
from . import LQLossFH
from config import device


class RobotsLoss(LQLossFH):
    def __init__(
        self, ybar, Q, alpha_u=1,
        alpha_col=None, alpha_obst=None, alpha_terminal=None,
        loss_bound=None, sat_bound=None,
        n_agents=1, min_dist=0.5,
        obstacle_centers=None, obstacle_covs=None
    ):
        super().__init__(Q=Q, R=alpha_u, loss_bound=loss_bound, sat_bound=sat_bound, ybar=ybar)
        self.n_agents = n_agents
        self.alpha_col, self.alpha_obst, self.alpha_terminal, self.min_dist = alpha_col, alpha_obst, alpha_terminal, min_dist
        assert (self.alpha_col is None and self.min_dist is None) or not (self.alpha_col is None or self.min_dist is None)
        if self.alpha_col is not None:
            assert self.n_agents is not None
        if self.alpha_terminal==0:
            self.alpha_terminal = None
        if not self.alpha_terminal is None:
            assert self.alpha_terminal >= 1
        # define obstacles
        if self.alpha_obst is not None:
            if obstacle_centers is None:
                if self.n_agents==2:
                    self.obstacle_centers = [
                        torch.tensor([[-2.5, 0]], device=device),
                        torch.tensor([[2.5, 0.0]], device=device),
                        torch.tensor([[-1.5, 0.0]], device=device),
                        torch.tensor([[1.5, 0.0]], device=device),
                    ]
                elif self.n_agents==1:
                    self.obstacle_centers = [
                        torch.tensor([[1, -1]], device=device),
                        # torch.tensor([[1.25, -0.75]], device=device),
                    ]
            else:
                self.obstacle_centers = obstacle_centers
            if obstacle_covs is None:
                self.obstacle_covs = [
                    torch.tensor([[0.2, 0.2]], device=device)
                ] * len(self.obstacle_centers)
            else:
                self.obstacle_covs = obstacle_covs
        else:
            self.obstacle_centers = None
            self.obstacle_covs = None

        # mask
        self.mask = torch.logical_not(torch.eye(self.n_agents, device=device))   # shape = (n_agents, n_agents)

    def forward(self, ys, us):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, input_dim)

        Return:
            - loss of shape (1, 1).
        """
        # batch
        y_batch = ys.reshape(*ys.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        # loss states = 1/T sum_{t=1}^T (y_t-ybar)^T Q (y_t-ybar)
        if self.ybar is not None:
            y_batch_centered = y_batch - self.ybar
        else:
            y_batch_centered = y_batch
        yTQy = torch.matmul(
            torch.matmul(y_batch_centered.transpose(-1, -2), self.Q),
            y_batch_centered
        )   # shape = (S, T, 1, 1)
        loss_y = torch.sum(yTQy, 1) / y_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )   # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / y_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        # terminal cost
        if self.alpha_terminal is None:
            loss_terminal = 0
        else:
            loss_terminal = torch.matmul(
                y_batch_centered[:, -1, :, :].transpose(-1, -2),
                y_batch_centered[:, -1, :, :]
            )   # shape = (S, 1, 1)
        # collision avoidance loss
        if self.alpha_col is None:
            loss_ca = 0
        else:
            loss_ca = self.alpha_col * self.f_loss_ca(y_batch)       # shape = (S, 1, 1)
        # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(y_batch) # shape = (S, 1, 1)
        # sum up all losses
        loss_val = loss_y + loss_u + loss_terminal + loss_ca + loss_obst            # shape = (S, 1, 1)
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/ys.shape[0]       # shape = (1, 1)
        return loss_val

    def f_loss_obst(self, y_batched):
        """
        Obstacle avoidance loss.

        Args:
            - y_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - obstacle avoidance loss of shape (1, 1).
        """
        q_horizontal = y_batched[:, :, 0::int(y_batched.shape[-2]/self.n_agents), :]   # position x of all agents. shape = (S, T, n_agents, 1)
        q_vertical = y_batched[:, :, 1::int(y_batched.shape[-2]/self.n_agents), :]   # position y of all agents. shape = (S, T, n_agents, 1)
        # batched over all samples and all times of [x agent 1, y agent 1, ..., x agent n, y agent n]
        q = torch.cat((q_horizontal,q_vertical), dim=-1).view(y_batched.shape[0], y_batched.shape[1], 1,-1).squeeze(dim=2)    # shape = (S, T, 2*n_agents)
        # sum up loss due to each obstacle #TODO
        for ind, (center, cov) in enumerate(zip(self.obstacle_centers, self.obstacle_covs)):
            if ind == 0:
                loss_obst = normpdf(q, mu=center, cov=cov)   # shape = (S, T)
            else:
                loss_obst += normpdf(q, mu=center, cov=cov)  # shape = (S, T)
        # average over time steps
        loss_obst = loss_obst.sum(1) / loss_obst.shape[1]    # shape = (S)
        return loss_obst.reshape(-1, 1, 1)

    def f_loss_ca(self, y_batch):
        """
        Collision avoidance loss.

        Args:
            - y_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.


        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = self.min_dist + 0.2
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(y_batch)              # shape = (S, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * self.mask).sum((-1, -2))/2        # shape = (S, T)
        # average over time steps
        loss_ca = loss_ca.sum(1)/loss_ca.shape[1]
        # reshape to S,1,1
        loss_ca = loss_ca.reshape(-1,1,1)
        return loss_ca

    def count_collisions(self, y_batch):
        """
        Count the number of collisions between agents.

        Args:
            - y_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - number of collisions between agents.
        """
        if len(y_batch.shape) == 3:
            y_batch = y_batch.reshape(*y_batch.shape, 1)
        distance_sq = self.get_pairwise_distance_sq(y_batch)  # shape = (S, T, n_agents, n_agents)
        col_matrix = (0.0001 < distance_sq) * (distance_sq < self.min_dist ** 2)  # Boolean collision matrix of shape (S, T, n_agents, n_agents)
        n_coll = col_matrix.sum().item()    # all collisions at all times and across all rollouts
        return n_coll/2                     # each collision is counted twice

    def get_pairwise_distance_sq(self, y_batch):
        """
        Squared distance between pairwise agents.

        Args:
            - y_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.

        Return:
            - matrix of shape (S, T, n_agents, n_agents) of squared pairwise distances.
        """
        state_dim_per_agent = int(y_batch.shape[2]/self.n_agents)
        # collision avoidance:
        x_agents = y_batch[:, :, 0::state_dim_per_agent, :]  # start from 0, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        y_agents = y_batch[:, :, 1::state_dim_per_agent, :]  # start from 1, pick every state_dim_per_agent. shape = (S, T, n_agents, 1)
        deltaqx = x_agents.repeat(1, 1, 1, self.n_agents) - x_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        deltaqy = y_agents.repeat(1, 1, 1, self.n_agents) - y_agents.repeat(1, 1, 1, self.n_agents).transpose(-2, -1)   # shape = (S, T, n_agents, n_agents)
        distance_sq = deltaqx ** 2 + deltaqy ** 2             # shape = (S, T, n_agents, n_agents)
        return distance_sq

    def __call__(self, x, u):
        return self.forward(x, u)  # Ensures the object is callable

def normpdf(q, mu, cov):  #TODO
    """
    PDF of normal distribution with mean "mu" and covariance "cov".

    Args:
        - q: shape(S, T, state_dim_per_agent)
        - mu:
        - cov:

    Return:
            -
    """
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2, dim=-1)
    for ind, qi in enumerate(qs):
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        nom = torch.exp((-0.5 * (qi - mu)**2 / cov).sum(-1))
        # out = out + num/den
        if ind == 0:
            out = nom/den
        else:
            out += nom/den
    return out

