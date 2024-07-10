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

    args = parser.parse_args()

    return args


def print_args(args):
    raise NotImplementedError