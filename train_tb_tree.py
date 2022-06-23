import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from utils import add_args, setup, set_seed, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='TB-Tree-based GFlowNet for hypergrid environment')
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

    exp_name = 'tbt_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    model.to(device)
    log_Z = nn.Parameter(torch.zeros((1,), device=device))

    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': 0.001}, {'params': [log_Z], 'lr': 0.1}]
    )

    total_loss = []
    total_reward = []
    total_l1_error = []
    total_visited_states = []

    for step in range(1, args.num_steps + 1):

        optimizer.zero_grad()

        loss_TB = torch.zeros((bsz,), device=device)
        loss_TB += log_Z

        # initial state: s_0 = [0, 0]
        states = torch.zeros((bsz, n), dtype=torch.long, device=device)

        # initial done trajectories: False
        dones = torch.full((bsz,), False, dtype=torch.bool, device=device)

        actions = None

        while torch.any(~dones):

            # ~dones: non-dones
            non_done_states = states[~dones]

            # Output: [logits_PF, logits_PB], where for logits_PF, the last position corresponds to 'stop' action.
            # logits: [non_done_bsz, 2 * n + 1]
            outputs = model(get_one_hot(non_done_states, h).view(non_done_states.size(0), -1))

            # Use the previous chosen actions, because PB is calculated on the same trajectory as PF.
            if actions is not None:
                loss_TB[~dones] -= 0.

            # Forward policy
            # edge_mask: We can't exceed the edge coordinates, e.g., (H - 1, xxx), (xxx, H - 1), (H - 1, H - 1)
            # stop_mask: any state can be a terminal state, so we append a 1 at the end
            logits_PF = outputs[:, :n + 1]
            prob_mask = get_mask(non_done_states, h)
            log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
            sampling_probs = (
                    (1. - args.random_action_prob) * (log_ProbF / args.temp).softmax(1)
                    + args.random_action_prob * (1 - prob_mask) / (1 - prob_mask + 1e-20).sum(1).unsqueeze(1)
            )
            actions = sampling_probs.multinomial(1)

            loss_TB[~dones] += log_ProbF.gather(dim=1, index=actions).squeeze(1)

            terminates = (actions.squeeze(-1) == n)

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            with torch.no_grad():
                non_terminates = actions[~terminates]
                states[~dones] = states[~dones].scatter_add(
                    1, non_terminates, torch.ones(non_terminates.size(), dtype=torch.long, device=device)
                )

        R = get_rewards(states, h, R0)
        loss = (loss_TB - R.log()).pow(2).mean()

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_reward.append(R.mean().item())

        if step % 100 == 0:
            emp_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            emp_density /= emp_density.sum()
            l1 = np.abs(true_density - emp_density).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info(
                'Step: %d, \tLoss: %.5f, \tlogZ: %.5f, \tR: %.5f, \tL1: %.5f' % (
                    step, np.array(total_loss[-100:]).mean(), log_Z.item(), np.array(total_reward[-100:]).mean(), l1
                )
            )

    pickle.dump(
        [total_loss, total_reward, total_visited_states, first_visited_states, total_l1_error],
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
