import os, sys, time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
plt.rcParams['axes.grid'] = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from controllers.contractive_ren import ContractiveREN
from loss_functions import RobotsLoss
from assistive_functions import WrapLogger
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
                   y_target=dataset.y_target,
                   x_init=dataset.x_init,
                   u_init=None, # zero
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   ).to(device)
sys_copy = RobotsSystem(input_noise_std=args.input_noise_std,
                   output_noise_std=args.output_noise_std,
                   y_target=dataset.y_target,
                   x_init=dataset.x_init,
                   u_init=None, # zero
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   ).to(device)
sys_copy.x_init = sys_copy.x_init*1.1


#--------- primal step dataset------------
# OPTION 1: use different initial conditions
# num_init_conditions = 10
# std_init_conditions = 0.2
# init_conditions_train_full = sys.x_init.repeat(num_init_conditions, 1, 1) + std_init_conditions * torch.randn(num_init_conditions, 1, sys.state_dim, device=sys.x_init.device)
# init_conditions_test = sys.x_init.repeat(100, 1, 1) + std_init_conditions * torch.randn(100, 1, sys.state_dim, device=sys.x_init.device)
# # validation data
# if args.early_stopping or args.return_best:
#     valid_inds = torch.randperm(init_conditions_train_full.shape[0])[:int(args.validation_frac*init_conditions_train_full.shape[0])]
#     train_inds = [ind for ind in range(init_conditions_train_full.shape[0]) if ind not in valid_inds]
#     init_conditions_valid = init_conditions_train_full[valid_inds, :, :]
#     init_conditions_train = init_conditions_train_full[train_inds, :, :]
# else:
#     init_conditions_valid = None
#     init_conditions_train = init_conditions_train_full

# OPTION 2: same inital condition, only noise on the output
output_noise_train_full = sys.generate_output_noise(
    num_samples=args.num_rollouts_K, horizon=args.horizon,
    noise_only_on_init=False
).to(device)
output_noise_test = sys.generate_output_noise(
    num_samples=500, horizon=args.horizon,
    noise_only_on_init=False
).to(device)
# validation data
if args.early_stopping or args.return_best_K:
    valid_inds = torch.randperm(output_noise_train_full.shape[0])[:int(args.validation_frac*output_noise_train_full.shape[0])]
    train_inds = [ind for ind in range(output_noise_train_full.shape[0]) if ind not in valid_inds]
    output_noise_valid = output_noise_train_full[valid_inds, :, :] if len(valid_inds)>0 else None
    output_noise_train = output_noise_train_full[train_inds, :, :]
else:
    output_noise_valid = None
    output_noise_train = output_noise_train_full
# # data for plots
# t_ext = args.horizon * 4
# plot_data = torch.zeros(1, t_ext, sys.state_dim, device=device)
# plot_data[:, 0, :] = sys.x_init.detach()
# plot_data = plot_data.to(device)
plot_data = output_noise_test[0:1, :, :]
# batch the data
train_dataloader = DataLoader(
    output_noise_train,
    batch_size=min(args.batch_size_K, args.num_rollouts_K), shuffle=True
)

# ------------ 3. Controller ------------
output_amplification = 20    # TODO: Note that this used to be 20!
logger.info('output_amplification for K0 = '+ str(output_amplification))
logger.info('[Info] internal model is sys')
K0 = PerfBoostController(internal_model=sys_copy, # G0, #
                          input_init=sys.y_init_nominal,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std,
                          output_amplification=output_amplification,
                          ren_internal_state_init = None, # TODO,
                          ).to(device)

# print(K0.internal_model.X.requires_grad)
total_n_params = sum(p.numel() for p in K0.parameters() if p.requires_grad)
logger.info("[INFO] Number of parameters: %i" % total_n_params)

# ------------ 4. Loss primal ------------
Q = torch.kron(torch.eye(args.n_agents), torch.eye(2)).to(device)   # TODO: move to args and print info
loss_fn_primal = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u, ybar=sys.y_target,
    loss_bound=None, sat_bound=None,
    alpha_terminal=args.alpha_terminal,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=args.n_agents,
)
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
y_log, _, u_log = sys.rollout(K0, plot_data)
filename = 'CL_init'
plot_trajectories(
    y_log[0, :, :],  # remove extra dim due to batching
    save_folder=save_folder, ybar=sys.y_target, n_agents=sys.n_agents,
    filename=filename, text="CL - before training", T=plot_data.shape[1],
    obstacle_centers=loss_fn_primal.obstacle_centers if args.obst_av else None,
    obstacle_covs=loss_fn_primal.obstacle_covs if args.obst_av else None,
)
plot_traj_vs_time(
    n_agents=sys.n_agents, y=y_log[0, :args.horizon, :],
    u=u_log[0, :args.horizon, :], save_folder=save_folder,
    filename=filename
)
# plot open loop
y_log_openloop= sys.openloop_rollout(u=None, output_noise=plot_data)
filename = 'openloop'
plot_trajectories(
    y_log_openloop[0, :, :],  # remove extra dim due to batching
    save_folder=save_folder, ybar=sys.y_target, n_agents=sys.n_agents,
    filename=filename, text="Open Loop", T=plot_data.shape[1],
    obstacle_centers=loss_fn_primal.obstacle_centers if args.obst_av else None,
    obstacle_covs=loss_fn_primal.obstacle_covs if args.obst_av else None,
)
plot_traj_vs_time(
    n_agents=sys.n_agents, y=y_log_openloop[0, :args.horizon, :],
    save_folder=save_folder, filename=filename
)
# ------------ 5. Training primal ------------
K0.fit(
    sys=sys, train_dataloader=train_dataloader, valid_data=output_noise_valid,
    lr=args.lr_K, loss_fn=loss_fn_primal, epochs=args.epochs_K, log_epoch=args.log_epoch_K,
    return_best=args.return_best_K, logger=logger, early_stopping=args.early_stopping,
    n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage,
    save_folder=save_folder, plot_data=plot_data
)
# ------ 6. Save and evaluate the trained model ------
# save
res_dict = K0.emme.state_dict()
# TODO: append args
res_dict['Q'] = Q
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % output_noise_train_full.shape[0])
with torch.no_grad():
    y_log, _, u_log = sys.rollout(
        controller=K0, output_noise_data=output_noise_train_full, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn_primal.forward(y_log, u_log)
    msg = 'Loss: %.4f' % loss
# count collisions
if args.col_av:
    num_col = loss_fn_primal.count_collisions(y_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % output_noise_test.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    y_log, _, u_log = sys.rollout(
        controller=K0, output_noise_data=output_noise_test, train=False,
    )
    # loss
    test_loss = loss_fn_primal.forward(y_log, u_log).item()
    msg = "Loss: %.4f" % test_loss
# count collisions
if args.col_av:
    num_col = loss_fn_primal.count_collisions(y_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
y_log, _, u_log = sys.rollout(K0, plot_data)
filename = 'CL_trained'
plot_trajectories(
    y_log[0, :, :],  # remove extra dim due to batching
    sys.y_target, args.n_agents, filename=filename, text="CL - trained controller", T=plot_data.shape[1],
    save_folder=save_folder,
    obstacle_centers=loss_fn_primal.obstacle_centers if args.obst_av else None,
    obstacle_covs=loss_fn_primal.obstacle_covs if args.obst_av else None,
)
plot_traj_vs_time(
    n_agents=sys.n_agents, y=y_log[0, :args.horizon, :], u=u_log[0, :args.horizon, :],
    save_folder=save_folder, filename=filename)

filename = os.path.join(save_folder, 'y_log'+'.pt')
torch.save(y_log, filename)

filename = os.path.join(save_folder, 'u_log'+'.pt')
torch.save(u_log, filename)

