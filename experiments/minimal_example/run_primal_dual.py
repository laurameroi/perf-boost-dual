import os, sys, time
import logging
import torch
import numpy as np
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
from controllers.contractive_ren import ContractiveREN
from loss_functions import RobotsLoss
from assistive_functions import WrapLogger
import math
from argparse import Namespace
# from controllers.SSMs import DWN, DWNConfig

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
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, n_agents=1)

sys = RobotsSystem(input_noise_std=args.input_noise_std,
                   output_noise_std=args.output_noise_std,
                   x_target=dataset.x_target,
                   x_init=dataset.x_init,
                   u_init=None, # zero
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   ).to(device)

# ------------ Open loop data collection ------------
openloop_data_out, openloop_data_in = dataset.generate_openloop_dataset(
    num_samples=args.num_samples_sysid, ts=0.05, noise_only_on_init=False, sys=sys
)
# NOTE: initial condition is always fixed and there's no noise on it.

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for n in range(args.num_samples_sysid):
    axs[0].plot(openloop_data_in[n, :, 0].detach().cpu().numpy())
    axs[1].plot(openloop_data_out[n, :, 0].detach().cpu().numpy())
axs[0].set_title('noisy reference data to the plant')
axs[1].set_title('noisy output data of the plant')
plt.savefig('openloop_data.png')

#-----------------------Dual initial step: learn G0 from open loop data--------------------------
#SSM with parallel scan
# set up a simple architecture
'''
cfg = {
    "n_u": 2,
    "n_y": 4,
    "d_model": 5,
    "d_state": 5,
    "n_layers": 3,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": False,
    "trainable": False,
    "gain": 2.4
}
cfg = Namespace(**cfg)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)

G0 = DWN(cfg.n_u, cfg.n_y, config)

# Define the loss function
loss_fn_dual = torch.nn.MSELoss()

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(G0.parameters(), lr=args.lr)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(args.epochs)

for epoch in range(args.epochs):
    optimizer.zero_grad()  # Reset gradients

    # Forward pass through the SSM
    ySSM, _ = G0(openloop_data_in, state=None, mode="scan")
    ySSM = torch.squeeze(ySSM)  # Remove unnecessary dimensions

    # Calculate the mean squared error loss
    loss = loss_fn_dual(ySSM, openloop_data_out)
    loss.backward()  # Backpropagate to compute gradients
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss
# Plot the training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(np.arange(args.epochs), LOSS, label='Training Loss', color='blue')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Plot some example predictions vs true values
n_example = 5  # Number of example trajectories to plot

plt.figure(figsize=(12, 8))
for i in range(min(n_example, openloop_data_in.shape[0])):
    true_values = openloop_data_out[i, :, :].detach().numpy()  # True output (shape: T, 4)
    predicted_values = ySSM[i, :, :].detach().numpy()  # Predicted output from the model (shape: T, 4)

    # Plot true vs predicted for each example
    plt.subplot(min(n_example, 5), 1, i + 1)
    plt.plot(true_values[:, 0], label='True Output - 1st dimension')
    plt.plot(predicted_values[:, 0], label='Predicted Output - 1st dimension', linestyle='--')
    plt.title(f'Example {i + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

#validation
# Forward pass through the SSM for validation data
ySSM_val, _ = G0(openloop_data_in_val, state=None, mode="scan")
yval = torch.squeeze(yval)

# Compute validation loss
loss_val = MSE(ySSM_val, openloop_data_out_val)
'''

# Create the model G0
G0 = ContractiveREN(
    input_dim=sys.input_dim, output_dim=sys.output_dim,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl, 
    y_init=sys.y_init.detach().clone()   # initialize the hidden state of REN s.t. initial REN output close to the nominal initial output of the plant
).to(device)

# plot before training
n_example = 5  # Number of example trajectories to plot
plt.figure(figsize=(10, 6))
for n in range(min(n_example, openloop_data_in.shape[0])):
    true_values = openloop_data_out[n, :, :].detach().cpu().numpy()  # True output (shape: T, 4)
    predicted_values = np.zeros_like(true_values)

    G0.reset()  # Reset model for the trajectory
    for t in range(openloop_data_in.shape[1]):
        u = openloop_data_in[n, t, :].unsqueeze(0)  # Input for time t
        u = u.unsqueeze(1)  # Shape (1, 1, 2)
        y_hat = G0(u)  # Forward pass
        predicted_values[t, :] = y_hat.squeeze().detach().cpu().numpy()

    # Plot true vs predicted for each example
    plt.subplot(min(n_example, 5), 1, n + 1)
    plt.plot(true_values[:, 0], label='True Output - 1st dimension')
    plt.plot(predicted_values[:, 0], label='Predicted Output - 1st dimension', linestyle='--')
    plt.title(f'Example {n + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Value')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.grid(True)

plt.tight_layout()
plt.savefig('G0_init')


# Define the optimizer and learning rate
optimizer = torch.optim.Adam(G0.parameters(), lr=args.lr)

# Define the loss function
loss_fn_dual = torch.nn.MSELoss()

# Training loop settings
LOSS = np.zeros(args.epochs)

start_time = time.time()
for epoch in range(args.epochs):
    total_loss = 0.0  # Initialize loss for reporting

    # Training loop
    for n in range(openloop_data_in.shape[0]):  # Loop over trajectories
        G0.reset()  # Reset model state
        loss = 0.0  # Reset loss for this trajectory

        for t in range(openloop_data_in.shape[1]):  # Loop over time steps
            u = openloop_data_in[n, t, :].unsqueeze(0).unsqueeze(1)  # (1,1,2)
            y = openloop_data_out[n, t, :].unsqueeze(0).unsqueeze(1)  # (1,1,4)

            y_hat = G0(u)  # Forward pass
            loss += loss_fn_dual(y, y_hat)  # Accumulate loss over time

        # Backpropagate after the entire trajectory
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients before next trajectory

        total_loss += loss.item()  # Accumulate for reporting

    LOSS[epoch] = total_loss

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {LOSS[epoch]} \t||\t Elapsed Time: {time.time()-start_time}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(np.arange(args.epochs), LOSS, label='Training Loss', color='blue')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
plt.savefig('loss')

# Plot some example predictions vs true values

plt.figure(figsize=(10, 6))
for n in range(min(n_example, openloop_data_in.shape[0])):
    true_values = openloop_data_out[n, :, :].detach().cpu().numpy()  # True output (shape: T, 4)
    predicted_values = np.zeros_like(true_values)

    G0.reset()  # Reset model for the trajectory
    for t in range(openloop_data_in.shape[1]):
        u = openloop_data_in[n, t, :].unsqueeze(0)  # Input for time t
        u = u.unsqueeze(1)  # Shape (1, 1, 2)
        y_hat = G0(u)  # Forward pass
        predicted_values[t, :] = y_hat.squeeze().detach().cpu().numpy()

    # Plot true vs predicted for each example
    plt.subplot(min(n_example, 5), 1, n + 1)
    plt.plot(true_values[:, 0], label='True Output - 1st dimension')
    plt.plot(predicted_values[:, 0], label='Predicted Output - 1st dimension', linestyle='--')
    plt.title(f'Example {n + 1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Value')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.grid(True)

plt.tight_layout()
plt.savefig('G0_trained')
exit()
#--------- primal step dataset------------
train_data_full = sys.generate_output_noise(num_samples = args.num_rollouts, horizon = args.horizon, noise_only_on_init=True, output_noise_std=0.1)
test_data = sys.generate_input_noise(num_samples = args.num_rollouts, horizon = args.horizon, noise_only_on_init=True, input_noise_std=0.1)
train_data_full, test_data = train_data_full.to(device), test_data.to(device)
# validation data
if args.early_stopping or args.return_best:
    valid_inds = torch.randperm(train_data_full.shape[0])[:int(args.validation_frac*train_data_full.shape[0])]
    train_inds = [ind for ind in range(train_data_full.shape[0]) if ind not in valid_inds]
    valid_data = train_data_full[valid_inds, :, :]
    train_data = train_data_full[train_inds, :, :]
else:
    valid_data = None
    train_data = train_data_full
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
plot_data[:, 0, :] = (sys.x_init.detach() - sys.x_target)
plot_data = plot_data.to(device)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


# ------------ 3. Controller ------------
ctl = PerfBoostController(internal_model=G0,
                          input_init=sys.y_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std,
                          output_amplification=1,  # TODO: Note that this used to be 20!
                          ren_internal_state_init = None # TODO
                          ).to(device)

# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
filename = os.path.join(save_folder, 'CL_init.pdf')
plot_trajectories(
    x_log[0, :, :],  # remove extra dim due to batching
    sys.x_target, sys.n_agents, filename=filename, text="CL - before training", T=t_ext
)
plot_traj_vs_time(args.horizon, sys.n_agents, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], save=False)
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
logger.info("[INFO] Number of parameters: %i" % total_n_params)

# ------------ 4. Loss primal ------------
Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
loss_fn_primal = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u, xbar=sys.x_target,
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=args.n_agents,
)
# ------------ 5. Training primal ------------

ctl.fit(
    sys=sys, train_dataloader=train_dataloader, valid_data=valid_data,
    lr=args.lr, loss_fn=loss_fn_primal, epochs=args.epochs, log_epoch=args.log_epoch,
    return_best=args.return_best, logger=logger, early_stopping=args.early_stopping,
    n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage
)

# ------ 6. Save and evaluate the trained model ------
# save
res_dict = ctl.emme.state_dict()
# TODO: append args
res_dict['Q'] = Q
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn_primal.forward(x_log, u_log)
    msg = 'Loss: %.4f' % loss
# count collisions
if args.col_av:
    num_col = loss_fn_primal.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    test_loss = loss_fn_primal.forward(x_log, u_log).item()
    msg = "Loss: %.4f" % test_loss
# count collisions
if args.col_av:
    num_col = loss_fn_primal.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
filename = os.path.join(save_folder, 'CL_trained.pdf')
plot_trajectories(
    x_log[0, :, :],  # remove extra dim due to batching
    sys.x_target, args.n_agents, filename=filename, text="CL - trained controller", T=t_ext,
    obstacle_centers=loss_fn_primal.obstacle_centers,
    obstacle_covs=loss_fn_primal.obstacle_covs
)
plot_traj_vs_time(args.horizon, sys.n_agents, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], save=False)


# ---- Closed-loop data collection-----
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_noise=args.std_noise, n_agents=1)
# divide to train and test
train_data_x, train_data_u = dataset.generate_closedloop_dataset(sys=sys, controller=ctl, noise_only_on_init=True, num_samples=50, ts=0.05)

#-------------Dual step: learn G1 from closed-loop data-----------------

nominal = False
init_k = 1.0 if nominal else 0.5 # true value is 1.0
init_b = 1.0 if nominal else 1.8 # true value is 1.0
init_mass = 1.0 if nominal else 0.1 # true value is 1.0
init_b2 = None if args.linearize_plant else 0.1  # None for linearized_plant, 0.1 o.w.
G1 = TrainableRobotsSystem(x_target=sys.x_target,
                   x_init=sys.x_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   init_k=init_k, init_b=init_b, init_mass=init_mass, init_b2=init_b2
                   ).to(device)

optimizer = torch.optim.Adam(G1.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    optimizer.zero_grad()
    x_log = G0.openloop_rollout(u=openloop_data_in, noise=None, train=True)
    loss = loss_fn_primal(x_log, openloop_data_out)
    loss.backward()
    optimizer.step()
    if epoch%args.log_epoch == 0:
        print('Epoch: ', epoch, 'Loss: ', loss.item())
        print([p.item() for p in G0.parameters()])

# Save the trained model's parameters
torch.save(G0.state_dict(), "G0_trained.pth")
# Define the loss function
MSE = nn.MSELoss()

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(Qg.parameters(), lr=args.lr)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

# Start training timer
t0 = time.time()
validation_losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0.0
    # Training loop
    for n in range(input_data_training.shape[0]):
        for t in range(input_data_training.shape[1]):
            if t == 0:
                u_K = torch.zeros(2)
                ctl.reset()
                state = None
            u_ext = input_data_training[n, t, :]
            u = u_ext - u_K
            u = u.view(1, 1, 2)  # Reshape input
            y_hat, state = Qg(u, state=state, mode="loop")
            y_hat = y_hat.squeeze(0).squeeze(0)
            u_K = ctl(y_hat)
            loss = loss + MSE(output_data_training[n, t, :], y_hat[:])
            y_hat_train[n, t, :] = y_hat.detach()
