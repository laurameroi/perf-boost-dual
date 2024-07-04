
from plots import *

# data for plot
t_ext = t_end * 4
plot_data = torch.zeros(t_ext, train_data.shape[-1])
plot_data[0, :] = (x0.detach() - xbar)
plot_data = plot_data.to(device)