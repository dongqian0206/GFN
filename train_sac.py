import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from collections import defaultdict
from itertools import count
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='SAC for hypergrid environment')
    parser.add_argument(
        '--sac_alpha', type=float, default=0.98 * np.log(1 / 3)
    )
    parser.add_argument(
        '--tau', type=float, default=0.
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
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    model.to(device)

    Qm1 = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    Qm1.to(device)

    Qm2 = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    Qm2.to(device)

    Qt1 = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    Qt1.to(device)

    Qt2 = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    Qt2.to(device)

    learned_alpha = torch.tensor([args.sac_alpha], requires_grad=True, device=device)

    optimizer = optim.Adam(
        params=list(model.parameters()) + list(Qm1.parameters()) + list(Qm2.parameters()) + [learned_alpha], lr=0.0001
    )

    total_loss = []
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
                logits = model(get_one_hot(non_done_states, h))
            prob_mask = get_mask(non_done_states, h)
            log_probs = torch.log_softmax(logits - 1e10 * prob_mask, -1)
            actions = log_probs.softmax(1).multinomial(1)

            induced_states = non_done_states + 0
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    induced_states[i, action] += 1

            terminates = (actions.squeeze(-1) == n)

            # Update batches
            c = count(0)
            m = {j: next(c) for j in range(bsz) if not dones[j]}
            for (i, _), ps, pa, s, t in zip(
                    sorted(m.items()), non_done_states, actions, induced_states, terminates.float()
            ):
                rs = get_rewards(s, h, R0) if t else torch.tensor(0., device=device)
                trajectories[i].append([ps.view(1, -1), pa.view(1, -1), s.view(1, -1), rs.view(-1), t.view(1, -1)])

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = induced_states[~terminates]

        parent_states, parent_actions, induced_states, rewards, finishes = [
            torch.cat(i) for i in zip(*sum(trajectories.values(), []))
        ]

        parent_one_hot = get_one_hot(parent_states, h)
        parent_mask = get_mask(parent_states, h)

        parent_log_probs = torch.log_softmax(model(parent_one_hot) - 1e10 * parent_mask, -1)
        parent_probs = parent_log_probs.exp()

        Qm1_logits = Qm1(parent_one_hot) * (1 - parent_mask)
        Qm1_sa = Qm1_logits.gather(dim=1, index=parent_actions).squeeze(1)

        Qm2_logits = Qm2(parent_one_hot) * (1 - parent_mask)
        Qm2_sa = Qm2_logits.gather(dim=1, index=parent_actions).squeeze(1)

        with torch.no_grad():
            children_one_hot = get_one_hot(induced_states, h)
            children_mask = get_mask(induced_states, h)
            children_log_probs = torch.log_softmax(model(children_one_hot) - 1e10 * children_mask, -1)
            children_probs = children_log_probs.exp()
            Qt1_logits = Qt1(children_one_hot) * (1 - children_mask)
            Qt2_logits = Qt2(children_one_hot) * (1 - children_mask)

        values1 = ((1 - finishes) * children_probs * (Qt1_logits - learned_alpha * children_log_probs)).sum(1)
        values2 = ((1 - finishes) * children_probs * (Qt2_logits - learned_alpha * children_log_probs)).sum(1)

        J_Qs = 0.5 * ((Qm1_sa - rewards - values1).pow(2) + (Qm2_sa - rewards - values2).pow(2)).mean()
        minQ = torch.min(Qm1_logits, Qm2_logits).detach()
        J_pi = (parent_probs * (learned_alpha * parent_log_probs - minQ)).sum(1).mean()
        J_alpha = (parent_probs.detach() * (args.sac_alpha - learned_alpha * parent_log_probs.detach())).sum(1).mean()

        for A, B in [(Qm1, Qt1), (Qm2, Qt2)]:
            for a, b in zip(A.parameters(), B.parameters()):
                b.data.mul_(1. - args.tau).add_(args.tau * a)

        loss = J_Qs + J_pi + J_alpha

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_reward.append(get_rewards(states, h, R0).mean().item())

        if step % 100 == 0:
            emp_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            emp_density /= emp_density.sum()
            l1 = np.abs(true_density - emp_density).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info(
                'Step: %d, \tLoss: %.5f, \tR: %.5f, \tL1: %.5f' % (
                    step, np.array(total_loss[-100:]).mean(), np.array(total_reward[-100:]).mean(), l1
                )
            )

    pickle.dump(
        [total_loss, total_reward, total_visited_states, first_visited_states, total_l1_error],
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
