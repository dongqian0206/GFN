import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random
import os


device = torch.device('cuda')


def add_args(parser):
    parser.add_argument(
        '--n', type=int, default=2
    )
    parser.add_argument(
        '--h', type=int, default=64
    )
    parser.add_argument(
        '--R0', type=float, default=1e-3
    )
    parser.add_argument(
        '--R1', type=float, default=0.5
    )
    parser.add_argument(
        '--R2', type=float, default=2
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3
    )
    parser.add_argument(
        '--hidden_size', type=int, default=256
    )
    parser.add_argument(
        '--num_layers', type=int, default=2
    )
    parser.add_argument(
        '--num_steps', type=int, default=62500
    )
    parser.add_argument(
        '--bsz', type=int, default=16
    )
    parser.add_argument(
        '--temp', type=float, default=1
    )
    parser.add_argument(
        '--epsilon', type=float, default=0
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    args = parser.parse_args()
    return args


def set_exp_name(exp_name, args):
    exp_name += '_{}'.format(args.seed)
    return exp_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(exp_name, args):
    exp_path = set_exp_name(exp_name, args)
    os.makedirs(exp_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(exp_path, 'log.txt'),
        filemode='w',
        format='%(asctime)s - %(levelname)s -  %(message)s',
        datefmt='%Y-%m-%d_%H-%M-%S',
        level=logging.INFO
    )
    logger.info(args)
    return logger, exp_path


def make_grid(n, h):
    grid = torch.zeros((h,) * n + (n,), dtype=torch.long, device=device)
    for i in range(n):
        grid_i = torch.linspace(start=0, end=h - 1, steps=h)
        for _ in range(i):
            grid_i = grid_i.unsqueeze(1)
        grid[..., i] = grid_i
    grid = grid.view(h ** n, -1)
    return grid


def get_rewards(x, h, R0, R1=0.5, R2=2):
    ax = abs(x / (h - 1) - 0.5)
    return R0 + R1 * (0.25 < ax).prod(-1) + R2 * ((0.3 < ax) * (ax < 0.4)).prod(-1)


def make_model(num_units, tail=None):
    if tail is None:
        tail = []
    layers = [[nn.Linear(i, o)] + ([nn.LeakyReLU()] if n < len(num_units[:-1]) - 1 else [])
              for n, (i, o) in enumerate(zip(num_units[:-1], num_units[1:]))]
    return nn.Sequential(*(sum(layers, []) + tail))


def get_one_hot(s, h):
    return F.one_hot(s, h).view(s.size(0), -1).float()


def get_mask(states, h, is_backward=False):
    # edge_mask: We can't exceed the edge coordinates, e.g., (H - 1, xxx), (xxx, H - 1), (H - 1, H - 1)
    # stop_mask: any state can be a terminal state, so we append a 1 at the end
    if is_backward:
        return (states == 0).float()
    edge_mask = (states == h - 1).float()
    stop_mask = torch.zeros((states.size(0), 1), device=device)
    prob_mask = torch.cat([edge_mask, stop_mask], 1)
    return prob_mask


def get_modes_found(first_visited_states, modes, num_steps):
    first_visited_states = torch.from_numpy(first_visited_states)[modes].long()
    mode_founds = (0 <= first_visited_states) & (first_visited_states <= torch.arange(1, num_steps + 1).unsqueeze(1))
    return mode_founds.sum(1)
