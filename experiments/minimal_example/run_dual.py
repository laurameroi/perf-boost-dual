import os, sys, copy
import logging
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, TrainableRobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss
from assistive_functions import WrapLogger


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

args = argument_parser()

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results')
save_folder = os.path.join(save_path, 'perf_boost_'+args.nn_type+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_'+args.nn_type+'_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Plant ------------
num_samples=50
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_noise=args.std_noise, n_agents=1)

plant_input_init = None     # all zero

sys = RobotsSystem(x_target=dataset.x_target,
                   x_init=dataset.x_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   ).to(device)

# ------------ Open loop data collection ------------
openloop_data_out, openloop_data_in = dataset.generate_openloop_dataset(
    num_samples=num_samples, ts=0.05, noise_only_on_init=True, sys=sys
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for n in range(num_samples):
    axs[0].plot(openloop_data_out[n, :, 0].detach().cpu().numpy())
    axs[1].plot(openloop_data_in[n, :, 0].detach().cpu().numpy())  
plt.savefig('foo.png')

# Learn G0
nominal = False
init_k = 1.0 if nominal else 0.5 # true value is 1.0
init_b = 1.0 if nominal else 1.8 # true value is 1.0
init_mass = 1.0 if nominal else 0.1 # true value is 1.0
init_b2 = None if args.linearize_plant else 0.1  # None for linearized_plant, 0.1 o.w.
G0 = TrainableRobotsSystem(x_target=dataset.x_target,
                   x_init=sys.x_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   init_k=init_k, init_b=init_b, init_mass=init_mass, init_b2=init_b2
                   ).to(device)
torch.save(G0.state_dict(), "G0_initial.pth")
optimizer = torch.optim.Adam(G0.parameters(), lr=args.lr)
# compute loss
loss_fn = torch.nn.MSELoss()

for epoch in range(args.epochs):
    optimizer.zero_grad()
    x_log = G0.openloop_rollout(u=openloop_data_in, output_noise=None, train=True)
    loss = loss_fn(x_log, openloop_data_out)
    loss.backward()
    optimizer.step()
    if epoch%args.log_epoch == 0:
        print('Epoch: ', epoch, 'Loss: ', loss.item())
        print([p.item() for p in G0.parameters()])

# Save the trained model's parameters
torch.save(G0.state_dict(), "G0_trained.pth")