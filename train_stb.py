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
    parser = argparse.ArgumentParser(description='SubTB-based GFlowNet for hypergrid environment')
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

    exp_name = 'stb_{}_{}_{}_{}'.format(n, h, R0, args.uniform_PB)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
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

            terminates = (actions.squeeze(-1) == n)

            # Update batches
            c = count(0)
            m = {j: next(c) for j in range(bsz) if not dones[j]}
            for (i, _), ps, pa, s, t in zip(
                    sorted(m.items()), non_done_states, actions, induced_states, terminates.float()
            ):
                trajectories[i].append([ps.view(1, -1), pa.view(1, -1), s.view(1, -1), t.view(-1)])

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = induced_states[~terminates]

        # Consecutive states
        parent_states, parent_actions, induced_states, finishes = [
            torch.cat(i) for i in zip(*[traj for traj in sum(trajectories.values(), [])])
        ]
        log_rewards = get_rewards(states, h, R0).log().view(-1, 1)

        idxs = iter(range(parent_states.size(0)))
        lens = [len(traj) for traj in trajectories.values()]
        batch_idxs = [torch.LongTensor(list(islice(idxs, i))).to(device) for i in lens]

        mask = trajectories.values()

        outputs = model(get_one_hot(parent_states, h))
        log_flowF, logits_PF = outputs[:, 2 * n + 1], outputs[:, :n + 1]
        prob_mask = get_mask(parent_states, h)
        log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
        log_ProbF = log_ProbF.gather(dim=1, index=parent_actions).squeeze(1)

        outputs = model(get_one_hot(induced_states, h))
        log_flowB, logits_PB = outputs[:, 2 * n + 1], outputs[:, n + 1:2 * n + 1]
        logits_PB = (0 if args.uniform_PB else 1) * logits_PB
        edge_mask = get_mask(induced_states, h, is_backward=True)
        log_ProbB = torch.log_softmax(logits_PB - 1e10 * edge_mask, -1)
        log_ProbB = torch.cat([log_ProbB, torch.zeros((log_ProbB.size(0), 1), device=device)], 1)
        log_ProbB = log_ProbB.gather(dim=1, index=parent_actions).squeeze(1)

        # log F(s_{0}) + \sum_{t=0}^{l} log P_F(s_{t+1} | s_{t}), for l = 0,...,n, s_{0} -- the initial state
        log_fwd_probs = torch.cat(
            [log_ProbF.index_select(0, i).cumsum(0) for i in batch_idxs]
        )
        loss_fwd_pf = log_flowF[0] + log_fwd_probs

        # log F(s_{l+1}) + \sum_{t=0}^{l} log P_B(s_{t} | s_{t+1}), for l = 0,...,n
        log_bwd_probs = torch.cat(
            [log_ProbB.index_select(0, i).cumsum(0) for i in batch_idxs]
        )
        loss_fwd_pb = log_flowB * (1 - finishes) + log_rewards * finishes + log_bwd_probs

        loss_fwd = (loss_fwd_pf - loss_fwd_pb).pow(2).mean()

        # log F(s_{l}) + \sum_{t=l}^{n} log P_F(s_{t+1} | s_{t}), for l = 0,...,n
        log_fwd_probs = torch.cat(
            [log_ProbF.index_select(0, i).flip(0).cumsum(0).flip(0) for i in batch_idxs]
        )
        loss_fwd_pf = log_flowF + log_fwd_probs

        # log R(x) + \sum_{t=l}^{n-1} log P_B(s_{t} | s_{t+1}), for l = 0,...,n
        loss_fwd_pb = torch.cat(
            [log_ProbB.index_select(0, i).flip(0).cumsum(0).flip(0) + log_rewards[r] for r, i in enumerate(batch_idxs)]
        )

        loss_bwd = (loss_fwd_pf - loss_fwd_pb).pow(2).mean()

        loss = loss_fwd + loss_bwd

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
