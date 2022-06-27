import torch
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards


def get_train_args():
    parser = argparse.ArgumentParser(description='Metropolis–Hastings for hypergrid environment')
    return parser


def main():
    device = torch.device('cuda')

    parser = get_train_args()
    args = add_args(parser)

    set_seed(args.seed)

    n = args.n
    h = args.h
    R0 = args.R0
    bsz = args.bsz

    exp_name = 'mcmc_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    total_l1_error = []
    total_visited_states = []

    # initial state: s_0 = [0, 0]
    states = torch.zeros((bsz, n), dtype=torch.long, device=device)

    for step in range(1, args.num_steps + 1):

        pre_rewards = get_rewards(states, h, R0)

        actions = torch.randint(0, n * 2, (bsz,), device=device)

        induced_states = states + 0
        for i, action in enumerate(actions.squeeze(-1)):
            if action < n:
                induced_states[i, action] = min(induced_states[i, action] + 1, h - 1)
            if action >= n:
                induced_states[i, action - n] = max(induced_states[i, action - n] - 1, 0)

        cur_rewards = get_rewards(induced_states, h, R0)

        A = cur_rewards / pre_rewards
        U = torch.rand((bsz,), device=device)

        updates = (A > U)

        for state in induced_states[updates]:
            state_id = (state * coordinate).sum().item()
            total_visited_states.append(state_id)
            if first_visited_states[state_id] < 0:
                first_visited_states[state_id] = step

        # Update non-updated trajectories
        states[updates] = induced_states[updates]

        if step % 100 == 0:
            emp_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            emp_density /= emp_density.sum()
            l1 = np.abs(true_density - emp_density).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info('Step: %d, \tL1: %.5f' % (step, l1))

    pickle.dump(
        [total_visited_states, first_visited_states, total_l1_error], open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
