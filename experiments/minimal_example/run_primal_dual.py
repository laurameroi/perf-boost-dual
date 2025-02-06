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
openloop_data_out_test, openloop_data_in_test = dataset.generate_openloop_dataset(
    num_samples=100, ts=0.05, noise_only_on_init=False, sys=sys
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for n in range(args.num_samples_sysid):
    axs[0].plot(openloop_data_in[n, :, 0].detach().cpu().numpy())
    axs[1].plot(openloop_data_out[n, :, 0].detach().cpu().numpy())
axs[0].set_title('noisy reference data to the plant')
axs[1].set_title('noisy output data of the plant')
plt.savefig('openloop_data.png')

#-----------------------Dual initial step: learn G0 from open loop data--------------------------
# Create the model G0
G0 = ContractiveREN(
    input_dim=sys.input_dim, output_dim=sys.output_dim,
    dim_internal=args.dim_internal, dim_nl=args.dim_nl, 
    y_init=sys.y_init_nominal.detach().clone()   # initialize the hidden state of REN s.t. initial REN output close to the nominal initial output of the plant
).to(device)

# plot before training
n_example = 2  # Number of example trajectories to plot
plt.figure(figsize=(10, 6))
true_values = openloop_data_out_test[range(n_example), :, :].detach().cpu().numpy()
G0.reset()  # Reset model for the trajectory
predicted_values = G0.rollout(openloop_data_in_test[range(n_example), :, :]).detach().cpu().numpy()
# Plot true vs predicted for each example
fig, axs =plt.subplots(min(n_example, 5), 1)
for n in range(n_example):
    axs[n].plot(true_values[n, :, 0], label='True Output - 1st dimension')
    axs[n].plot(predicted_values[n, :, 0], label='Predicted Output - 1st dimension', linestyle='--')
    axs[n].set_title(f'Example on test data {n + 1}')
    axs[n].set_xlabel('Time Steps')
    axs[n].set_ylabel('Output Value')
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

#-----------training--------------
start_time = time.time()
for epoch in range(args.epochs):
    total_loss = 0.0  # Initialize loss for reporting
    G0.reset()  # Reset model state
    predicted_values = G0.rollout(openloop_data_in)
    loss = loss_fn_dual(openloop_data_out, predicted_values)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()  # Reset gradients before next trajectory

    LOSS[epoch] = loss

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {LOSS[epoch]} \t||\t Elapsed Time: {time.time()-start_time}")

with torch.no_grad():
    G0.reset()
    loss_test = loss_fn_dual(openloop_data_out_test, G0.rollout(openloop_data_in_test))
print('loss test', loss_test)


#-------------plots----------------
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
# plot before training
n_example = 2  # Number of example trajectories to plot
plt.figure(figsize=(10, 6))
true_values = openloop_data_out_test[range(n_example), :, :].detach().cpu().numpy()
G0.reset()  # Reset model for the trajectory
predicted_values = G0.rollout(openloop_data_in_test[range(n_example), :, :]).detach().cpu().numpy()
# Plot true vs predicted for each example
fig, axs =plt.subplots(min(n_example, 5), 1)
for n in range(n_example):
    axs[n].plot(true_values[n, :, 0], label='True Output - 1st dimension')
    axs[n].plot(predicted_values[n, :, 0], label='Predicted Output - 1st dimension', linestyle='--')
    axs[n].set_title(f'Example on test data {n + 1}')
    axs[n].set_xlabel('Time Steps')
    axs[n].set_ylabel('Output Value')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
plt.grid(True)

plt.tight_layout()
plt.savefig('G0_trained')


#--------- primal step dataset------------
# TODO: train and valid division
num_init_conditions = 10
std_init_conditions = 0.2
init_conditions_train_full = sys.x_init.repeat(num_init_conditions, 1, 1) + std_init_conditions * torch.randn(num_init_conditions, 1, sys.state_dim, device=sys.x_init.device)
init_conditions_test = sys.x_init.repeat(100, 1, 1) + std_init_conditions * torch.randn(100, 1, sys.state_dim, device=sys.x_init.device)

# validation data
if args.early_stopping or args.return_best:
    valid_inds = torch.randperm(init_conditions_train_full.shape[0])[:int(args.validation_frac*init_conditions_train_full.shape[0])]
    train_inds = [ind for ind in range(init_conditions_train_full.shape[0]) if ind not in valid_inds]
    init_conditions_valid = init_conditions_train_full[valid_inds, :, :]
    init_conditions_train = init_conditions_train_full[train_inds, :, :]
else:
    init_conditions_valid = None
    init_conditions_train = init_conditions_train_full

# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, init_conditions_train.shape[-1], device=device)
plot_data[:, 0, :] = (sys.x_init.detach() - sys.x_target)
plot_data = plot_data.to(device)
# batch the data
train_dataloader = DataLoader(init_conditions_train, batch_size=args.batch_size, shuffle=True)


# ------------ 3. Controller ------------
ctl = PerfBoostController(internal_model=G0,
                          input_init=sys.y_init_nominal,
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
    sys=sys, train_dataloader=train_dataloader, valid_data=init_conditions_valid,
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
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % init_conditions_train.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=init_conditions_train, train=False,
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
