import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from collections import defaultdict
from itertools import count, islice
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='SubTB-lambda-based GFlowNets for hypergrid environments')
    parser.add_argument(
        '--uniform_PB', type=int, choices=[0, 1], default=0
    )
    parser.add_argument(
        '--gamma', type=float, default=0.9
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
    R1 = args.R1
    R2 = args.R2
    bsz = args.bsz

    exp_name = 'SubTB_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        n, h, R0, args.lr, args.uniform_PB, args.gamma, args.temp, args.epsilon
    )
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0, R1, R2)
    modes = true_rewards.view(-1) >= true_rewards.max()
    num_modes = modes.sum().item()

    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [2 * n + 2])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    total_loss = []

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

            # Outputs: [logits_PF, logits_PB, logF]
            with torch.no_grad():
                outputs = model(get_one_hot(non_done_states, h))

            # Forward policy
            prob_mask = get_mask(non_done_states, h)
            log_probs = torch.log_softmax(outputs[:, :n + 1] - 1e10 * prob_mask, -1)

            temp_probs = (log_probs / args.temp).softmax(1)
            uniform_probs = (1 - prob_mask) / (1 - prob_mask).sum(dim=-1, keepdim=True)
            sampling_probs = (1 - args.epsilon) * temp_probs + args.epsilon * uniform_probs

            actions = sampling_probs.multinomial(1)

            child_states = non_done_states + 0
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    child_states[i, action] += 1

            terminates = (actions.squeeze(-1) == n)

            # Update batches
            c = count(0)
            m = {j: next(c) for j in range(bsz) if not dones[j]}
            for (i, _), ps, pa, s, t in zip(
                    sorted(m.items()), non_done_states, actions, child_states, terminates.float()
            ):
                lrs = get_rewards(s, h, R0, R1, R2).log() if t else torch.tensor(0., device=device)
                trajectories[i].append([ps.view(1, -1), pa.view(1, -1), s.view(1, -1), lrs.view(-1), t.view(-1)])

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = child_states[~terminates]

        parent_states, parent_actions, child_states, log_rewards, finishes = [
            torch.cat(i) for i in zip(*[traj for traj in sum(trajectories.values(), [])])
        ]

        # transition_ids = [[0, 1, 2, 3, 4], [5, 6, 7], [8], ...]
        traj_lens = [len(traj) for traj in trajectories.values()]
        ids = iter(range(parent_states.size(0)))
        transition_ids = [torch.LongTensor(list(islice(ids, i))).to(device) for i in traj_lens]

        p_outputs = model(get_one_hot(parent_states, h))
        log_flowF, logits_PF = p_outputs[:, 2 * n + 1], p_outputs[:, :n + 1]
        prob_mask = get_mask(parent_states, h)
        log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
        log_PF_sa = log_ProbF.gather(dim=1, index=parent_actions).squeeze(1)

        c_outputs = model(get_one_hot(child_states, h))
        log_flowB, logits_PB = c_outputs[:, 2 * n + 1], c_outputs[:, n + 1:2 * n + 1]
        logits_PB = (0 if args.uniform_PB else 1) * logits_PB
        edge_mask = get_mask(child_states, h, is_backward=True)
        log_ProbB = torch.log_softmax(logits_PB - 1e10 * edge_mask, -1)
        log_ProbB = torch.cat([log_ProbB, torch.zeros((log_ProbB.size(0), 1), device=device)], 1)
        log_PB_sa = log_ProbB.gather(dim=1, index=parent_actions).squeeze(1)
        log_flowB = log_flowB * (1 - finishes) + log_rewards * finishes

        loss = 0.

        for i, traj_len in enumerate(traj_lens):
            idx = torch.arange(1, traj_len + 1)
            weights = ((args.gamma ** idx) * (traj_len + 1 - idx)).sum()
            for sub_traj_len in range(1, traj_len + 1):
                traj_ids = torch.arange(sub_traj_len).tile(traj_len - sub_traj_len + 1, 1)
                batch_idx = transition_ids[i][traj_ids + torch.arange(traj_ids.size(0)).view(-1, 1)]
                loss_trajF = log_PF_sa[batch_idx].sum(-1) + log_flowF[batch_idx][:, 0]
                loss_trajB = log_PB_sa[batch_idx].sum(-1) + log_flowB[batch_idx][:, -1]
                traj_len_weight = (args.gamma ** sub_traj_len) / weights
                sub_traj_loss = traj_len_weight * (loss_trajF - loss_trajB).pow(2).sum()
                loss = loss + sub_traj_loss

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if step % 100 == 0:
            empirical_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            l1 = np.abs(true_density - empirical_density / empirical_density.sum()).mean()
            total_l1_error.append((len(total_visited_states), l1))
            first_state_founds = torch.from_numpy(first_visited_states)[modes].long()
            mode_founds = (0 <= first_state_founds) & (first_state_founds <= step)
            logger.info(
                'Step: %d, \tLoss: %.5f, \tL1: %.5f, \t\tModes found: [%d/%d]' % (
                    step,
                    np.array(total_loss[-100:]).mean(),
                    l1,
                    mode_founds.sum().item(),
                    num_modes
                )
            )

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

    logger.info('Done.')


if __name__ == '__main__':
    main()
