import torch
from assistive_functions import to_tensor
import torch.nn as nn
from config import device


class LQLossFH():
    def __init__(self, Q, R, loss_bound=None, sat_bound=None, ybar=None):
        self.Q, self.R = Q, R
        self.Q = to_tensor(self.Q).to(device)
        self.R = to_tensor(self.R)
        if isinstance(self.R, torch.Tensor):     # cast to device if is not a scalar
            self.R = self.R.to(device)
        assert len(self.Q.shape) == 2 and self.Q.shape[0] == self.Q.shape[1]
        assert (not hasattr(self.R, "__len__")) or len(self.R.shape) == 2  # int or square matrix
        self.loss_bound, self.sat_bound = loss_bound, sat_bound
        if self.loss_bound is not None:
            assert self.sat_bound is not None
            self.loss_bound = to_tensor(self.loss_bound)
        if self.sat_bound is not None:
            assert self.loss_bound is not None
            self.sat_bound = to_tensor(self.sat_bound)
        self.ybar = ybar
        if self.ybar is not None:
            self.ybar = to_tensor(self.ybar).to(device)
            self.ybar = self.ybar.reshape(self.Q.shape[0], 1)

    def forward(self, ys, us):
        """
        compute loss
        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, input_dim)
        """
        if self.ybar is not None:
            ys = ys - self.ybar.repeat(ys.shape[0], 1, 1)
        # batch
        ys = ys.reshape(*ys.shape, 1)
        us = us.reshape(*us.shape, 1)
        # batched multiplication
        xTQx = torch.matmul(torch.matmul(ys.transpose(-1, -2), self.Q), ys)         # shape = (S, T, 1, 1)
        uTRu = torch.matmul(torch.matmul(us.transpose(-1, -2), self.R), us)         # shape = (S, T, 1, 1)
        # average over the time horizon
        loss_x = torch.sum(xTQx, 1) / ys.shape[1]    # shape = (S, 1, 1)
        loss_u = torch.sum(uTRu, 1) / us.shape[1]    # shape = (S, 1, 1)
        loss_val = loss_x + loss_u
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/ys.shape[0]       # shape = (1, 1)
        return loss_val
