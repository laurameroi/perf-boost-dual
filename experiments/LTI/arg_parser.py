import argparse, math


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="LTI experiment.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')

    # dataset
    parser.add_argument('--horizon', type=int, default=10, help='Time horizon for the computation. Default is 10.')
    parser.add_argument('--state-dim', type=int, default=1, help='Number of states of the LTI Plant. Default is 1.')
    parser.add_argument('--num-rollouts', type=int, default=30, help='Number of rollouts in the training data. Default is 30.')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=5, help='Number of forward trajectories of the closed-loop system at each step. Default is 5.')
    # optimizer - early stopping
    parser.add_argument('--early-stopping', type=str2bool, default=True, help='Stop SGD if validation loss does not significantly decrease.')
    parser.add_argument('--validation-frac', type=float, default=0.25, help='Fraction of data used for validation. Default is 0.25.')
    parser.add_argument('--n-logs-no-change', type=int, default=5, help='Early stopping if the validation loss does not improve by at least tol percentage during the last n_logs_no_change logged epochs. Default is 5.')
    parser.add_argument('--tol-percentage', type=float, default=0.05, help='Early stopping if the validation loss does not improve by at least tol percentage during the last n_logs_no_change logged epochs. Default is 0.05%.')
    
    args = parser.parse_args()

    return args


def print_args(args):
    raise NotImplementedError

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'T', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'F', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')