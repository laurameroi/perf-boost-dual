import torch
import numpy as np
from config import device

def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x

def check_data_dim(x, vec_dim):
    # make sure the first dimension is batch_size
    if len(x.shape)==len(vec_dim)+1:
        for d_ind, d in enumerate(vec_dim):
            if not d is None:
                assert x.shape[d_ind+1]==d, 'dimension mismatch in dim ' +str(d_ind) + 'required ' + str(d) + ', found '+str(x.shape[d_ind+1])
        return x
    elif len(x.shape)==len(vec_dim):
        for d_ind, d in enumerate(vec_dim):
            if not d is None:
                assert x.shape[d_ind]==d, 'dimension mismatch in dim ' +str(d_ind)
        return x.reshape(1, *x.shape)
    else:
        print(x.shape, vec_dim)
        raise Exception()

class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = not (logger == None)
        self.logger=logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)
