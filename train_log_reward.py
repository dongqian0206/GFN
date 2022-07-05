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
    parser = argparse.ArgumentParser(description='log-reward-based GFlowNet for hypergrid environment')
    parser.add_argument(
        '--uniform_PB', type=int, choices=[0, 1], default=1
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

    exp_name = 'lr_{}_{}_{}_{}'.format(n, h, R0, args.uniform_PB)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [2 * n + 1])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

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

            # Output: [logits_PF, logits_PB], where for logits_PF, the last position corresponds to 'stop' action.
            with torch.no_grad():
                outputs = model(get_one_hot(non_done_states, h))

            logits_PF = outputs[:, :n + 1]
            prob_mask = get_mask(non_done_states, h)
            log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
            actions = log_ProbF.softmax(1).multinomial(1)

            induced_states = non_done_states + 0
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    induced_states[i, action] += 1

            prs, nrs = get_rewards(non_done_states, h, R0), get_rewards(induced_states, h, R0)

            terminates = (actions.squeeze(-1) == n)

            # Update batches
            c = count(0)
            m = {j: next(c) for j in range(bsz) if not dones[j]}
            for (i, _), ps, pa, s, pr, nr, t in zip(
                    sorted(m.items()), non_done_states, actions, induced_states, prs, nrs, terminates.float()
            ):
                trajectories[i].append(
                    [ps.view(1, -1), pa.view(1, -1), s.view(1, -1), pr.view(-1), nr.view(-1), t.view(1, -1)]
                )

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = induced_states[~terminates]

        parent_states, parent_actions, induced_states, parent_rewards, induced_rewards, finishes = [
            torch.cat(i) for i in zip(*[traj for traj in sum(trajectories.values(), []) if not traj[-1]])
        ]

        # logR(s) + logP_F(s' | s) - logP_F(sf | s) --> logR(s) + log \pi_F(a | s) - log \pi_F(stop | s)
        log_ProbF = torch.log_softmax(
            model(get_one_hot(parent_states, h))[:, :n + 1] - 1e10 * get_mask(parent_states, h), -1
        )
        loss_pt = parent_rewards.log() + log_ProbF.gather(dim=1, index=parent_actions).squeeze(1) - log_ProbF[:, n]

        # logR(s') + logP_B(s | s') - logP_F(sf | s') --> logR(s') + log \pi_B(a | s') - log \pi_F(stop | s')
        induced_logits = model(get_one_hot(induced_states, h))
        log_ProbF = torch.log_softmax(
            induced_logits[:, :n + 1] - 1e10 * get_mask(induced_states, h), -1
        )
        logits_PB = induced_logits[:, n + 1:2 * n + 1]
        logits_PB = (0 if args.uniform_PB else 1) * logits_PB
        log_ProbB = torch.log_softmax(
            logits_PB - 1e10 * get_mask(induced_states, h, is_backward=True), -1
        )
        loss_nt = induced_rewards.log() + log_ProbB.gather(dim=1, index=parent_actions).squeeze(1) - log_ProbF[:, n]

        loss = (loss_pt - loss_nt).pow(2).mean()

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

    with open(os.path.join(exp_path, 'model.pt'), 'wb') as f:
        torch.save(model, f)

    pickle.dump(
        [total_loss, total_reward, total_visited_states, first_visited_states, total_l1_error],
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
