import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from collections import defaultdict
from itertools import count
from copy import deepcopy
from utils import add_args, setup, set_seed, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='SAC for hypergrid environment')
    parser.add_argument(
        '--sac_alpha', type=float, default=0.98 * np.log(1 / 3)
    )
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

    exp_name = 'sac_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    pol = make_model(
        [n * h] + [args.hidden_size] * args.num_layers + [n + 1]
    )
    pol.to(device)

    Q_1 = make_model(
        [n * h] + [args.hidden_size] * args.num_layers + [n + 1]
    )
    Q_1.to(device)

    Q_2 = make_model(
        [n * h] + [args.hidden_size] * args.num_layers + [n + 1]
    )
    Q_2.to(device)

    Qt1 = make_model(
        [n * h] + [args.hidden_size] * args.num_layers + [n + 1]
    )
    Qt1.to(device)

    Qt2 = make_model(
        [n * h] + [args.hidden_size] * args.num_layers + [n + 1]
    )
    Qt2.to(device)

    learned_alpha = torch.tensor([args.sac_alpha], requires_grad=True, device=device)

    optimizer = optim.Adam(
        params=list(pol.parameters()) + list(Q_1.parameters()) + list(Q_2.parameters()) + [learned_alpha], lr=0.0001
    )

    total_loss = []
    total_J_Qs = []
    total_J_pi = []
    total_J_alpha = []
    total_reward = []
    total_l1_error = []
    total_visited_states = []

    for step in range(1, args.num_steps + 1):

        optimizer.zero_grad()

        # initial state: s_0 = [0, 0]
        states = torch.zeros((bsz, n), dtype=torch.long, device=device)

        # initial done trajectories: False
        dones = torch.full((bsz,), False, dtype=torch.bool, device=device)

        trajectories = defaultdict(list)

        while torch.any(~dones):

            # ~dones: non-dones
            non_done_states = states[~dones]

            with torch.no_grad():
                outputs = pol(get_one_hot(non_done_states, h).view(non_done_states.size(0), -1))
            prob_mask = get_mask(non_done_states, h)
            log_probs = torch.log_softmax(outputs - 1e10 * prob_mask, -1)
            sampling_probs = (
                    (1. - args.random_action_prob) * (log_probs / args.temp).softmax(1)
                    + args.random_action_prob * (1 - prob_mask) / (1 - prob_mask + 1e-20).sum(1).unsqueeze(1)
            )
            actions = sampling_probs.multinomial(1)

            induced_states = deepcopy(non_done_states)
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    induced_states[i, action] += 1

            terminates = (actions.squeeze(-1) == n)

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update batches
            c = count(0)
            m = {j: next(c) for j in range(bsz) if not dones[j]}
            for (i, _), ps, pa, s, t in zip(
                    sorted(m.items()), non_done_states, actions, induced_states, terminates.float()
            ):
                rs = get_rewards(s, h, R0) if t else torch.tensor(0., device=device)
                trajectories[i].append([ps.view(1, -1), pa, s.view(1, -1), rs.view(-1), t.view(-1)])

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = induced_states[~terminates]

        parent_states, parent_actions, induced_states, rewards, finishes = [
            torch.cat(i) for i in zip(*sum(trajectories.values(), []))
        ]

        parent_one_hot = get_one_hot(parent_states, h).view(parent_states.size(0), -1)
        parent_mask = get_mask(parent_states, h)
        print(parent_states)
        print(parent_mask)
        print('=' * 100)

        parent_log_probs = torch.log_softmax(pol(parent_one_hot) - 1e10 * parent_mask, -1)
        parent_probs = parent_log_probs.exp()

        q1s = Q_1(parent_one_hot) * (1 - parent_mask)
        q1a = q1s.gather(dim=1, index=parent_actions.unsqueeze(-1)).squeeze(-1)

        q2s = Q_2(parent_one_hot) * (1 - parent_mask)
        q2a = q2s.gather(dim=1, index=parent_actions.unsqueeze(-1)).squeeze(-1)

        children_one_hot = get_one_hot(induced_states, h).view(induced_states.size(0), -1)
        children_mask = get_mask(induced_states, h)

        with torch.no_grad():
            children_log_probs = torch.log_softmax(pol(children_one_hot) - 1e10 * children_mask, -1)
            children_probs = children_log_probs.exp()
            qt1 = Qt1(children_one_hot) * (1 - children_mask)
            qt2 = Qt2(children_one_hot) * (1 - children_mask)

        vcs1 = ((1 - finishes.unsqueeze(1)) * children_probs * (qt1 - learned_alpha * children_log_probs)).sum(1)
        vcs2 = ((1 - finishes.unsqueeze(1)) * children_probs * (qt2 - learned_alpha * children_log_probs)).sum(1)

        J_Qs = (0.5 * (q1a - rewards - vcs1).pow(2) + 0.5 * (q2a - rewards - vcs2).pow(2)).mean()
        J_pi = (parent_probs * (learned_alpha * parent_log_probs - torch.min(q1s, q2s).detach())).sum(1).mean()
        J_alpha = (parent_probs.detach() * (args.sac_alpha - learned_alpha * parent_log_probs.detach())).sum(1).mean()

        R = get_rewards(states, h, R0)
        loss = J_Qs + J_pi + J_alpha

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_J_Qs.append(J_Qs.item())
        total_J_pi.append(J_pi.item())
        total_J_alpha.append(J_alpha.item())
        total_reward.append(R.mean().item())

        if step % 100 == 0:
            emp_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            emp_density /= emp_density.sum()
            l1 = np.abs(true_density - emp_density).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info(
                'Step: %d, \tLoss: %.5f, \tJ_Qs: %.5f, \tJ_pi: %.5f, \t\tJ_alpha: %.5f, \tR: %.5f, \tL1: %.5f' % (
                    step, np.array(total_loss[-100:]).mean(), np.array(total_J_Qs[-100:]).mean(),
                    np.array(total_J_pi[-100:]).mean(), np.array(total_J_alpha[-100:]).mean(),
                    np.array(total_reward[-100:]).mean(), l1
                )
            )

    pickle.dump(
        [total_loss, total_reward, total_visited_states, first_visited_states, total_l1_error],
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
