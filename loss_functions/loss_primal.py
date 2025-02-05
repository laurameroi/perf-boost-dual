import torch
from config import device


class LossPrimal():
    def __init__(self, xbar, Q):
        self.Q = Q
        self.xbar = xbar.unsqueeze(1)

    def forward(self, xs, us):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)
        if self.xbar is not None:
            x_batch_centered = x_batch - self.xbar
        else:
            x_batch_centered = x_batch
        xTQx = torch.matmul(
            torch.matmul(x_batch_centered.transpose(-1, -2), self.Q),
            x_batch_centered
        )   # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / x_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        loss_x = torch.sum(loss_x, 0) / xs.shape[0]
        return loss_x