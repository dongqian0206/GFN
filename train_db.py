import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='DB-based GFlowNet for hypergrid environment')
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

    exp_name = 'db_{}_{}_{}_{}'.format(n, h, R0, args.uniform_PB)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [2 * n + 2])
    model.to(device)
    log_Z = torch.zeros((1,), device=device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    total_loss = []
    total_reward = []
    total_l1_error = []
    total_visited_states = []

    for step in range(1, args.num_steps + 1):

        optimizer.zero_grad()

        loss_DB = torch.zeros((bsz, n * h), device=device)

        # initial state: s_0 = [0, 0]
        states = torch.zeros((bsz, n), dtype=torch.long, device=device)

        # initial done trajectories: False
        dones = torch.full((bsz,), False, dtype=torch.bool, device=device)

        actions = None

        i = 0

        while torch.any(~dones):

            # ~dones: non-dones
            non_done_states = states[~dones]

            # Complete output [logits_PF, logits_PB, log_F], where log_F would be predicted for DB loss
            outputs = model(get_one_hot(non_done_states, h))

            # log_F(s_0), ..., log_F(s_t)
            log_flow = outputs[:, 2 * n + 1]
            loss_DB[~dones, i] += log_flow
            if i > 0:
                loss_DB[~dones, i - 1] -= log_flow
            else:
                log_Z[:] = log_flow[0].item()

            # Backward policy, i.e., given an object, samples a plausible trajectory that leads to it.
            logits_PB = outputs[:, n + 1:2 * n + 1]
            logits_PB = (0 if args.uniform_PB else 1) * logits_PB
            edge_mask = get_mask(non_done_states, h, is_backward=True)
            log_ProbB = torch.log_softmax(logits_PB - 1e10 * edge_mask, -1)

            # Use the previous chosen actions, because PB is calculated on the same trajectory as PF.
            if actions is not None:
                loss_DB[~dones, i - 1] -= log_ProbB.gather(dim=1, index=actions[actions != n].unsqueeze(1)).squeeze(1)

            # Forward policy
            logits_PF = outputs[:, :n + 1]
            prob_mask = get_mask(non_done_states, h)
            log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
            actions = log_ProbF.softmax(1).multinomial(1)

            loss_DB[~dones, i] += log_ProbF.gather(dim=1, index=actions).squeeze(-1)

            terminates = (actions.squeeze(-1) == n)

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            termination_mask = ~dones
            termination_mask[~dones] &= terminates
            R = get_rewards(non_done_states[terminates], h, R0)
            loss_DB[termination_mask, i] -= R.log()

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            with torch.no_grad():
                non_terminates = actions[~terminates]
                states[~dones] = states[~dones].scatter_add(
                    1, non_terminates, torch.ones(non_terminates.size(), dtype=torch.long, device=device)
                )

            i += 1

        loss = loss_DB.pow(2).mean()

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
                'Step: %d, \tLoss: %.5f, \tlogZ: %.5f, \tR: %.5f, \tL1: %.5f' % (
                    step, np.array(total_loss[-100:]).mean(), log_Z.item(), np.array(total_reward[-100:]).mean(), l1
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
