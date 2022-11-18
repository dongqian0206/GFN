import torch
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='Random-search for hypergrid environment')
    return parser


def main():
    device = torch.device('cuda')

    parser = get_train_args()
    args = add_args(parser)

    set_seed(args.seed)

    n = args.n
    h = args.h
    R0 = args.R0
    R1 = args.R1
    R2 = args.R2
    bsz = args.bsz

    exp_name = 'random_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0, R1, R2)
    modes = true_rewards.view(-1) >= true_rewards.max()
    num_modes = modes.sum().item()

    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    total_l1_error = []
    total_visited_states = []

    for step in range(1, args.num_steps + 1):

        # initial state: s_0 = [0, 0]
        states = torch.zeros((bsz, n), dtype=torch.long, device=device)

        # initial done trajectories: False
        dones = torch.full((bsz,), False, dtype=torch.bool, device=device)

        while torch.any(~dones):

            # ~dones: non-dones
            non_done_states = states[~dones]

            prob_mask = get_mask(non_done_states, h)
            probs = (1 - prob_mask) / torch.sum(1 - prob_mask, dim=1, keepdim=True)
            actions = probs.multinomial(1)

            child_states = non_done_states + 0
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    child_states[i, action] += 1

            terminates = (actions.squeeze(-1) == n)

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = child_states[~terminates]

        if step % 100 == 0:
            empirical_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            l1 = np.abs(true_density - empirical_density / empirical_density.sum()).mean()
            total_l1_error.append((len(total_visited_states), l1))
            first_state_founds = torch.from_numpy(first_visited_states)[modes].long()
            mode_founds = (0 <= first_state_founds) & (first_state_founds <= step)
            logger.info(
                'Step: %d, \tL1: %.5f, \t\tModes found: [%d/%d]' % (
                    step,
                    l1,
                    mode_founds.sum().item(),
                    num_modes
                )
            )

    pickle.dump(
        {
            'total_visited_states': total_visited_states,
            'first_visited_states': first_visited_states,
            'num_visited_states_so_far': [a[0] for a in total_l1_error],
            'total_l1_error': [a[1] for a in total_l1_error]
        },
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
