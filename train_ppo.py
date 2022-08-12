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
    parser = argparse.ArgumentParser(description='PPO for hypergrid environment')
    parser.add_argument(
        '--data_collection_num_steps', type=int, default=16
    )
    parser.add_argument(
        '--ppo_num_steps', type=int, default=32
    )
    parser.add_argument(
        '--ppo_clip', type=float, default=0.2
    )
    parser.add_argument(
        '--ppo_entropy_coef', type=float, default=0.1
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

    exp_name = 'ppo_{}_{}_{}_{}'.format(n, h, R0, args.ppo_entropy_coef)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1 + 1])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

    total_loss = []
    total_l1_error = []
    total_visited_states = []

    for step in range(1, args.num_steps + 1):

        trajectories = []

        for _ in range(1, args.data_collection_num_steps + 1):

            # initial state: s_0 = [0, 0]
            states = torch.zeros((bsz, n), dtype=torch.long, device=device)

            # initial done trajectories: False
            dones = torch.full((bsz,), False, dtype=torch.bool, device=device)

            batches = defaultdict(list)

            while torch.any(~dones):

                # ~dones: non-dones
                non_done_states = states[~dones]

                with torch.no_grad():
                    logits = model(get_one_hot(non_done_states, h))[:, :-1]

                prob_mask = get_mask(non_done_states, h)
                log_probs = torch.log_softmax(logits - 1e10 * prob_mask, -1)
                actions = log_probs.softmax(1).multinomial(1)
                log_probs = log_probs.gather(dim=1, index=actions).squeeze(1)

                induced_states = non_done_states + 0
                for i, action in enumerate(actions.squeeze(-1)):
                    if action < n:
                        induced_states[i, action] += 1

                terminates = (actions.squeeze(-1) == n)

                # Update batches
                c = count(0)
                m = {j: next(c) for j in range(bsz) if not dones[j]}
                for (i, _), ps, pa, s, t, lp in zip(
                        sorted(m.items()), non_done_states, actions, induced_states, terminates.float(), log_probs
                ):
                    rs = get_rewards(s, h, R0) if t else torch.tensor(0., device=device)
                    batches[i].append(
                        [ps.view(1, -1), pa.view(1, -1), s.view(1, -1), rs.view(-1), t.view(-1), lp.view(-1)]
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

            # Compute advantages
            for tau in batches.values():
                parent_states, parent_actions, induced_states, rewards, finishes, log_probs = [
                    torch.cat(i) for i in zip(*tau)
                ]
                with torch.no_grad():
                    values_ps = model(get_one_hot(parent_states, h))[:, -1]
                    values_is = model(get_one_hot(induced_states, h))[:, -1]
                adv = rewards + values_is * (1. - finishes) - values_ps
                for i, A in zip(tau, adv):
                    i.append(rewards[-1].unsqueeze(0))
                    i.append(A.unsqueeze(0))

            trajectories += sum(batches.values(), [])

        for _ in range(1, args.ppo_num_steps + 1):

            optimizer.zero_grad()

            idxs = np.random.randint(0, len(trajectories), bsz)
            parent_states, parent_actions, induced_states, rewards, finishes, pre_log_probs, Gs, As = [
                torch.cat(i) for i in zip(*[trajectories[i] for i in idxs])
            ]
            outputs = model(get_one_hot(parent_states, h))
            logits, values = outputs[:, :-1], outputs[:, -1]
            prob_mask = get_mask(parent_states, h)
            log_probs = torch.log_softmax(logits - 1e10 * prob_mask, -1)
            cur_log_probs = log_probs.gather(dim=1, index=parent_actions).squeeze(1)
            ratio = torch.exp(cur_log_probs - pre_log_probs)

            val_loss = 0.5 * (Gs - values).pow(2).mean()
            act_loss = -torch.min(As * ratio, As * ratio.clamp(1. - args.ppo_clip, 1. + args.ppo_clip)).mean()
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

            loss = val_loss + act_loss - args.ppo_entropy_coef * entropy

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        if step % 100 == 0:
            empirical_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            l1 = np.abs(true_density - empirical_density / empirical_density.sum()).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info('Step: %d, \tLoss: %.5f, \tL1: %.5f' % (step, np.array(total_loss[-100:]).mean(), l1))

    with open(os.path.join(exp_path, 'model.pt'), 'wb') as f:
        torch.save(model, f)

    pickle.dump(
        {
            'total_loss': total_loss,
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
