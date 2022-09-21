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
    parser.add_argument(
        '--weight', type=float, default=1
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

    exp_name = 'stb_{}_{}_{}_{}_{}_{}'.format(n, h, R0, args.uniform_PB, args.lr, args.weight)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    modes = true_rewards.view(-1) >= true_rewards.max()
    num_modes = modes.sum().item()

    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1 + 1])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    total_loss = []
    total_edge_loss = []
    total_leaf_loss = []
    total_traj_loss = []

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
                lrs = get_rewards(s, h, R0).log() if t else torch.tensor(0., device=device)
                trajectories[i].append([ps.view(1, -1), pa.view(1, -1), s.view(1, -1), lrs.view(-1), t.view(-1)])

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
        parent_states, parent_actions, induced_states, log_rewards, finishes = [
            torch.cat(i) for i in zip(*[traj for traj in sum(trajectories.values(), [])])
        ]

        # batch_idxs = [[0, 1, 2, 3, 4], [5, 6, 7], [8], ...]
        idxs = iter(range(parent_states.size(0)))
        lens = [len(traj) for traj in trajectories.values()]
        batch_idxs = [torch.LongTensor(list(islice(idxs, i))).to(device) for i in lens]

        p_outputs = model(get_one_hot(parent_states, h))
        log_flowF, logits_PF = p_outputs[:, n + 1], p_outputs[:, :n + 1]
        prob_mask = get_mask(parent_states, h)
        log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
        log_PF_sa = log_ProbF.gather(dim=1, index=parent_actions).squeeze(1)

        i_outputs = model(get_one_hot(induced_states, h))
        log_flowB, logits_PB = i_outputs[:, n + 1], i_outputs[:, :n]
        logits_PB = (0 if args.uniform_PB else 1) * logits_PB
        edge_mask = get_mask(induced_states, h, is_backward=True)
        log_ProbB = torch.log_softmax(logits_PB - 1e10 * edge_mask, -1)
        log_ProbB = torch.cat([log_ProbB, torch.zeros((log_ProbB.size(0), 1), device=device)], 1)
        log_PB_sa = log_ProbB.gather(dim=1, index=parent_actions).squeeze(1)

        # log F(s_{0}) + \sum_{t=0}^{l} log P_F(s_{t+1} | s_{t}), for l = 0,...,n
        # log P_F(s1 | s0)
        # log P_F(s1 | s0) + log P_F(s2 | s1)
        # log P_F(s1 | s0) + log P_F(s2 | s1) + log P_F(sf | s2)
        log_fwd_ProbF = torch.cat(
            [log_PF_sa.index_select(0, i).cumsum(0) for i in batch_idxs]
        )

        # log F(s0) + log P_F(s1 | s0)
        # log F(s0) + log P_F(s1 | s0) + log P_F(s2 | s1)
        # log F(s0) + log P_F(s1 | s0) + log P_F(s2 | s1) + log P_F(sf | s2)
        loss_fwd_PF = log_flowF[0] + log_fwd_ProbF

        # log F(s_{l+1}) + \sum_{t=0}^{l} log P_B(s_{t} | s_{t+1}), for l = 0,...,n
        # log P_B(s0 | s1)
        # log P_B(s0 | s1) + log P_B(s1 | s2)
        # log P_B(s0 | s1) + log P_B(s1 | s2)
        log_fwd_ProbB = torch.cat(
            [log_PB_sa.index_select(0, i).cumsum(0) for i in batch_idxs]
        )
        # log F(s1) + log P_B(s0 | s1)
        # log F(s2) + log P_B(s0 | s1) + log P_B(s1 | s2)
        # log R(s2) + log P_B(s0 | s1) + log P_B(s1 | s2)
        loss_fwd_PB = log_flowB * (1 - finishes) + log_rewards + log_fwd_ProbB

        loss_fwd = (loss_fwd_PF - loss_fwd_PB).pow(2)

        # Mask the last trajectory balance loss and other losses given by one state
        mask_fwd = torch.LongTensor(sum([[1] * (l - 1) + [0] * 1 for i, l in enumerate(lens)], [])).to(device)
        loss_fwd = (loss_fwd * mask_fwd).sum() / mask_fwd.sum()

        log_PF_edge_flows = log_flowF + log_PF_sa
        log_PB_edge_flows = log_flowB * (1 - finishes) + log_rewards + log_PB_sa

        last_idxs = torch.LongTensor([i[-1] for i in batch_idxs]).to(device)
        log_Z_log_fwd_probs = loss_fwd_PF.index_select(0, last_idxs)
        log_R_log_bwd_probs = loss_fwd_PB.index_select(0, last_idxs)

        # log F(s_{l}) + \sum_{t=l}^{n} log P_F(s_{t+1} | s_{t}), for l = 0,...,n
        # log P_F(s1 | s0) + log P_F(s2 | s1) + log P_F(sf | s2)
        # log P_F(s2 | s1) + log P_F(sf | s2)
        # log P_F(sf | s2)
        log_bwd_ProbF = torch.cat(
            [log_PF_sa.index_select(0, i).flip(0).cumsum(0).flip(0) for i in batch_idxs]
        )
        # log F(s0) + log P_F(s1 | s0) + log P_F(s2 | s1) + log P_F(sf | s2)
        # log F(s1) + log P_F(s2 | s1) + log P_F(sf | s2)
        # log F(s2) + log P_F(sf | s2)
        loss_bwd_PF = log_flowF + log_bwd_ProbF

        # log R(x) + \sum_{t=l}^{n-1} log P_B(s_{t} | s_{t+1}), for l = 0,...,n
        # log P_B(s0 | s1) + log P_B(s1 | s2)
        # log P_B(s1 | s2)
        # 0
        log_bwd_ProbB = torch.cat(
            [log_PB_sa.index_select(0, i).flip(0).cumsum(0).flip(0) for i in batch_idxs]
        )
        # log R(s2) + log P_B(s0 | s1) + log P_B(s1 | s2)
        # log R(s2) + log P_B(s1 | s2)
        # log R(s2)
        log_rewards = log_rewards[finishes.bool()]
        loss_bwd_PB = torch.stack(sum([[log_rewards[i]] * l for i, l in enumerate(lens)], [])) + log_bwd_ProbB

        loss_bwd = (loss_bwd_PF - loss_bwd_PB).pow(2)
        loss_bwd = loss_bwd.mean()

        with torch.no_grad():
            edge_loss = ((log_PF_edge_flows - log_PB_edge_flows) * (1 - finishes)).pow(2).sum() / (1 - finishes).sum()
            leaf_loss = ((log_PF_edge_flows - log_PB_edge_flows) * finishes).pow(2).sum() / finishes.sum()
            traj_loss = (log_Z_log_fwd_probs - log_R_log_bwd_probs).pow(2).mean()

        loss = loss_fwd + args.weight * loss_bwd

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_edge_loss.append(edge_loss.item())
        total_leaf_loss.append(leaf_loss.item())
        total_traj_loss.append(traj_loss.item())

        if step % 100 == 0:
            empirical_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            l1 = np.abs(true_density - empirical_density / empirical_density.sum()).mean()
            total_l1_error.append((len(total_visited_states), l1))
            first_state_founds = torch.from_numpy(first_visited_states)[modes].long()
            mode_founds = (0 <= first_state_founds) & (first_state_founds <= step)
            logger.info(
                'Step: %d, \tLoss: %.5f, \tEdge_loss: %.5f, \tLeaf_loss: %.5f, \tTrajectory_loss: %.5f, '
                '\tL1: %.5f, \t\tModes found: [%d/%d]' % (
                    step,
                    np.array(total_loss[-100:]).mean(),
                    np.array(total_edge_loss[-100:]).mean(),
                    np.array(total_leaf_loss[-100:]).mean(),
                    np.array(total_traj_loss[-100:]).mean(),
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
            'total_edge_loss': total_edge_loss,
            'total_leaf_loss': total_leaf_loss,
            'total_traj_loss': total_traj_loss,
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
