import torch
from config import device


class LossPrimal():
    def __init__(self, ybar, Q):
        self.Q = Q
        self.ybar = ybar.unsqueeze(1)

    def forward(self, xs, us):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, input_dim)

        Return:
            - loss of shape (1, 1).
        """
        # batch
        y_batch = xs.reshape(*xs.shape, 1)
        if self.ybar is not None:
            y_batch_centered = y_batch - self.ybar
        else:
            y_batch_centered = y_batch
        xTQx = torch.matmul(
            torch.matmul(y_batch_centered.transpose(-1, -2), self.Q),
            y_batch_centered
        )   # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / y_batch.shape[1]    # average over the time horizon. shape = (S, 1, 1)
        loss_x = torch.sum(loss_x, 0) / xs.shape[0]
        return loss_x