import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import random
import os
from copy import deepcopy


device = torch.device('cuda')


def add_args(parser):
    parser.add_argument(
        '--n', type=int, default=2
    )
    parser.add_argument(
        '--h', type=int, default=64
    )
    parser.add_argument(
        '--R0', type=float, default=1e-2
    )
    parser.add_argument(
        '--hidden_size', type=int, default=256
    )
    parser.add_argument(
        '--num_layers', type=int, default=2
    )
    parser.add_argument(
        '--num_steps', type=int, default=50000
    )
    parser.add_argument(
        '--bsz', type=int, default=16
    )
    parser.add_argument(
        '--temp', type=float, default=1.
    )
    parser.add_argument(
        '--random_action_prob', type=float, default=0.
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    args = parser.parse_args()
    return args


def set_exp_name(exp_name, args):
    exp_name += '_{}'.format(args.seed)
    return exp_name


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(n, h):
    grid = torch.zeros((h,) * n + (n,), dtype=torch.long, device=device)
    for i in range(n):
        grid_i = torch.linspace(start=0, end=h - 1, steps=h)
        for _ in range(i):
            grid_i = grid_i.unsqueeze(1)
        grid[..., i] = grid_i
    grid = grid.view(h ** n, -1)
    return grid


def get_rewards(x, h, R0):
    ax = abs(x / (h - 1) - 0.5)
    return R0 + 0.5 * (0.25 < ax).prod(-1) + 2 * ((0.3 < ax) * (ax < 0.4)).prod(-1)


def make_model(num_units, tail=None):
    if tail is None:
        tail = []
    layers = [[nn.Linear(i, o)] + ([nn.LeakyReLU()] if n < len(num_units[:-1]) - 1 else [])
              for n, (i, o) in enumerate(zip(num_units[:-1], num_units[1:]))]
    return nn.Sequential(*(sum(layers, []) + tail))


def get_one_hot(s, h):
    return F.one_hot(s, h).float()


def get_mask(states, h, is_backward=False):
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


def plot_policy(model, grid, n, h, step, exp_path):
    num_grids = grid.size(0)
    model.eval()
    with torch.no_grad():
        logits = model(get_one_hot(grid, h).view(num_grids, -1))
        logits_PF, logits_PB = logits[:, :n + 1], logits[:, n + 1:2 * n + 1]
        log_ProbF = torch.log_softmax(
            logits_PF - 1e10 * torch.cat([grid.eq(h - 1).float(), torch.zeros((num_grids, 1), device=device)], 1), -1
        )
        log_ProbB = torch.log_softmax(
            logits_PB - 1e10 * (grid == 0).float(), -1
        )
    grid = grid.cpu().numpy()
    ProbF = log_ProbF.exp().cpu().numpy()
    ProbB = log_ProbB.exp().cpu().numpy()
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for i in range(num_grids):
        for action in [0, 1]:
            if action == 0:
                dx = ProbF[i, action]
                dy = 0
            else:
                dx = 0
                dy = ProbF[i, action]
            axes[0].arrow(
                x=grid[i, 0], y=grid[i, 1], dx=dx, dy=dy, width=0.04, length_includes_head=True, ec='black', fc='black'
            )
    for i in range(num_grids):
        for action in [0, 1]:
            if action == 0:
                dx = -ProbB[i, action]
                dy = 0
            else:
                dx = 0
                dy = -ProbB[i, action]
            axes[1].arrow(
                x=grid[i, 0], y=grid[i, 1], dx=dx, dy=dy, width=0.04, length_includes_head=True, ec='black', fc='black'
            )
    axes[0].scatter(
        x=grid[:, 0], y=grid[:, 1], s=ProbF[:, 2] * 200, marker='8', color='r'
    )
    axes[0].set_title('Forward Policy')
    axes[1].set_title('Backward Policy')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'traj_{step}.png'), bbox_inches='tight', format='png', dpi=200)


def sample_trajectories(model, rewards, bsz, n, h, exp_path):
    model.eval()
    with torch.no_grad():
        states = torch.zeros((bsz, n), dtype=torch.long, device=device)
        dones = torch.full((bsz,), False, dtype=torch.bool, device=device)
        trajectories = [deepcopy(states).cpu().numpy()]
        while torch.any(~dones):
            # ~dones: non-dones
            num_non_dones = (~dones).sum().item()
            non_done_states = states[~dones]
            logits = model(F.one_hot(non_done_states, h).view(num_non_dones, -1).float())
            edge_mask = non_done_states.eq(h - 1).float()
            stop_mask = torch.zeros((num_non_dones, 1), device=device)
            prob_mask = torch.cat([edge_mask, stop_mask], 1)
            log_ProbF = torch.log_softmax(
                logits[..., :n + 1] - 1e10 * prob_mask, -1
            )
            actions = log_ProbF.exp().multinomial(1)
            terminates = (actions.squeeze(-1) == n)
            # Update dones
            dones[~dones] |= terminates
            # Update non-done trajectories
            with torch.no_grad():
                non_terminates = actions[~terminates]
                states[~dones] = states[~dones].scatter_add(
                    1, non_terminates, torch.ones(non_terminates.size(), dtype=torch.long, device=device)
                )
            trajectories.append(deepcopy(states).cpu().numpy())
        plt.figure()
        sns.heatmap(
            rewards.cpu().numpy(), cmap='Blues', linewidths=0.5, square=True, cbar=False
        )
        plt.scatter(x=0.5, y=0.5, s=100, marker='*', color='m')
        plt.gca().invert_yaxis()
        colors = ['m', 'g', 'r', 'c', 'k', 'b']
        non_empty_states = np.sum(trajectories[-1], -1) != 0
        for traj1, traj2 in zip(trajectories[:-1], trajectories[1:]):
            diff = traj2 - traj1
            for i in range(bsz):
                if non_empty_states[i]:
                    plt.arrow(
                        x=traj1[i, 0] + 0.5, y=traj1[i, 1] + 0.5, dx=diff[i, 0], dy=diff[i, 1],
                        width=0.04, ec=colors[i], fc=colors[i]
                    )
                    plt.scatter(
                        x=trajectories[-1][i, 0] + 0.5, y=trajectories[-1][i, 1] + 0.5, s=100, marker='o',
                        color=colors[i]
                    )
        plt.tight_layout()
        plt.savefig(os.path.join(exp_path, 'sample.png'), bbox_inches='tight', format='png', dpi=200)
