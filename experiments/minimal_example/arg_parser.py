import argparse, math


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Robots minimal experiment.")

    # experiment primal
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')
    parser.add_argument('--col-av', type=str2bool, default=False, help='Avoid collisions. Default is True.')
    parser.add_argument('--obst-av', type=str2bool, default=True, help='Avoid obstacles. Default is True.')

    # dataset primal
    parser.add_argument('--horizon', type=int, default=100, help='Time horizon for the computation. Default is 100.')
    parser.add_argument('--n-agents', type=int, default=1, help='Number of agents. Default is 1.')
    parser.add_argument('--num-rollouts-K', type=int, default=30, help='Number of rollouts in the training data. Default is 30.')
    parser.add_argument('--input-noise-std', type=float, default=0.0, help='std of the noise on the input to the plant (d). Default is 0.0.')
    parser.add_argument('--output-noise-std', type=float, default=0.02, help='std of the noise on the plant output (v). Default is 0.2.')

    # dataset dual
    parser.add_argument('--num-rollouts-G', type=int, default=50, help='Number of signals in the training data. Default is 50.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0 , help='Spring constant. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=str2bool, default=True, help='Linearize plant or not. Default is False.')

    # controller
    parser.add_argument('--nn-type', type=str, default='REN',
                        help='Type of the NN for operator Emme in controller. Options: REN or SSM. Default is REN')
    parser.add_argument('--non-linearity', type=str,
                        help='Type of non_linearity in SSMs. Options: MLP, coupling_layers, hamiltonian, tanh. '
                             'Default coupling_layers.')
    parser.add_argument('--cont-init-std', type=float, default=0.1,
                        help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8,
                        help='Dimension of the internal state of the controller. '
                             'Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')

    # loss primal
    parser.add_argument('--alpha-u', type=float, default=0.1/400, help='Weight of the loss due to control input "u". Default is 0.1/400.')  #TODO: 400 is output_amplification^2
    parser.add_argument('--alpha-terminal', type=float, default=100, help='Weight of the terminal cost. Default is 100.')
    parser.add_argument('--alpha-col', type=float, default=100, help='Weight of the collision avoidance loss. Default is 100 if "col-av" is True, else None.')
    parser.add_argument('--alpha-obst', type=float, default=10, help='Weight of the obstacle avoidance loss. Default is 10 if "obst-av" is True, else None.')
    parser.add_argument('--min-dist', type=float, default=1.0, help='TODO. Default is 1.0 if "col-av" is True, else None.')  #TODO: add help

    # optimizer primal (controller)
    parser.add_argument('--batch-size-K', type=int, default=5, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    parser.add_argument('--epochs-K', type=int, default=500, help='Total number of epochs for training the controller. Default is 5000 if collision avoidance, else 100.')
    parser.add_argument('--lr-K', type=float, default=1e-2, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--log-epoch-K', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best-K', type=str2bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')
    # optimizer dual (plant)
    parser.add_argument('--batch-size-G', type=int, default=5, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    parser.add_argument('--epochs-G', type=int, default=500, help='Total number of epochs for training the plant. Default is 5000 if collision avoidance, else 100.')
    parser.add_argument('--lr-G', type=float, default=1e-2, help='Learning rate. Default is 2e-3 if collision avoidance, else 5e-3.')
    parser.add_argument('--log-epoch-G', type=int, default=-1, help='Frequency of logging in epochs. Default is 0.1 * epochs.')
    parser.add_argument('--return-best-G', type=str2bool, default=True, help='Return the best model on the validation data among all logged iterations. The train data can be used instead of validation data. The Default is True.')
    # optimizer - early stopping
    parser.add_argument('--early-stopping', type=str2bool, default=True, help='Stop SGD if validation loss does not significantly decrease.')
    parser.add_argument('--validation-frac', type=float, default=0.25, help='Fraction of data used for validation. Default is 0.25.')
    parser.add_argument('--n-logs-no-change', type=int, default=5, help='Early stopping if the validation loss does not improve by at least tol percentage during the last n_logs_no_change logged epochs. Default is 5.')
    parser.add_argument('--tol-percentage', type=float, default=0.05, help='Early stopping if the validation loss does not improve by at least tol percentage during the last n_logs_no_change logged epochs. Default is 0.05%.')
    
    # TODO: add the following
    # parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    # parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    # parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    # # save/load args
    # parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')
    # parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')
    # parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')

    args = parser.parse_args()

    # set default non-linearity for SSMs
    if args.nn_type == "SSM":
        args.non_linearity = 'coupling_layers'

    # set default values that depend on other args
    if args.batch_size_K == -1:
        args.batch_size_K = args.num_rollouts_K  # use all train data
    # set default values that depend on other args
    if args.batch_size_G == -1:
        args.batch_size_G = args.num_rollouts_G  # use all train data

    if args.epochs_G == -1 or args.epochs_G is None:
        args.epochs_G = 50
    if args.epochs_K == -1 or args.epochs_K is None:
        args.epochs_K = 1000 if args.col_av else 50

    if args.lr_K == -1 or args.lr_K is None:
        args.lr_K = 2e-3 if args.col_av else 5e-3
    if args.lr_G == -1 or args.lr_G is None:
        args.lr_G = 2e-3

    if args.log_epoch_K == -1 or args.log_epoch_K is None:
        args.log_epoch_K = math.ceil(float(args.epochs_K)/10)
    if args.log_epoch_G == -1 or args.log_epoch_G is None:
        args.log_epoch_G = math.ceil(float(args.epochs_G)/10)

    # assertions and warning
    if not args.col_av:
        args.alpha_col = None
        args.min_dist = None
    if not args.obst_av:
        args.alpha_obst = None

    # if args.total_epochs < 10000:
    #     print(f'Minimum of 10000 epochs are required for proper training')

    if args.horizon > 100:
        print(f'Long horizons may be unnecessary and pose significant computation')

    if args.return_best_G or args.return_best_K:
        assert args.validation_frac > 0, 'validation fraction must be positive for return best.'
        assert args.validation_frac < 1, 'validation fraction must be less than 1 for return best.'
    if args.early_stopping:
        assert args.validation_frac > 0, 'validation fraction must be positive for early stopping.'
        assert args.validation_frac < 1, 'validation fraction must be less than 1 for early stopping.'

    return args


def print_args(args):
    msg = '\n[INFO] Dataset: n_agents: %i' % args.n_agents + ' -- num_rollouts_K: %i' % args.num_rollouts_K + ' -- num_rollouts_G: %i' % args.num_rollouts_G
    msg += ' -- input noise std: %.2f' % args.input_noise_std + ' -- output noise std: %.2f' % args.output_noise_std + ' -- time horizon: %i' % args.horizon

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const + ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller using %ss: dimension of the internal state: %i' % (args.nn_type, args.dim_internal)
    msg += ' -- dim_nl: %i' % args.dim_nl + ' -- cont_init_std: %.2f' % args.cont_init_std
    if args.nn_type == "SSM":
        msg += ' -- non_linearity: %s' % args.non_linearity

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u
    msg += ' -- alpha_terminal: %.f' % args.alpha_terminal
    msg += ' -- alpha_col: %.f' % args.alpha_col if args.col_av else ' -- no collision avoidance'
    msg += ' -- alpha_obst: %.1f' % args.alpha_obst if args.obst_av else ' -- no obstacle avoidance'

    msg += '\n[INFO] Optimizer: lr_G: %.2e' % args.lr_G + ' -- lr_K: %.2e' % args.lr_K
    msg += ' -- epochs plant: %i,' % args.epochs_G + ' -- epochs controller: %i,' % args.epochs_K
    msg += ' -- batch_size_G: %i,' % args.batch_size_G + ' -- batch_size_K: %i,' % args.batch_size_K
    msg += ' -- return best model for validation data among logged epochs for plant and controller: ' + str(args.return_best_G) + str(args.return_best_K)
    if args.early_stopping:
        msg += '\n Early stopping enabled with validation fraction: %.2f' % args.validation_frac
        msg += ' -- n_logs_no_change: %i' % args.n_logs_no_change + ' -- tol percentage: %.2f' % args.tol_percentage
    else:
        msg += '\n Early stopping disabled'

    return msg

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'T', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'F', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')