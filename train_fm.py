import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='FM-based GFlowNet for hypergrid environment')
    return parser


def get_parent_states(s, a, n):
    if a == n:
        return torch.cat([s.view(1, -1)]), torch.cat([a])
    parents = []
    actions = []
    for i in range(n):
        if s[i] > 0:
            ps = s + 0
            ps[i] -= 1
            parents += [ps.view(1, -1)]
            actions += [torch.tensor([i], device=s.device)]
    return torch.cat(parents), torch.cat(actions)


def main():
    device = torch.device('cuda')

    parser = get_train_args()
    args = add_args(parser)

    set_seed(args.seed)

    n = args.n
    h = args.h
    R0 = args.R0
    bsz = args.bsz

    exp_name = 'fm_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 1])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

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

        trajectories = []

        while torch.any(~dones):

            # ~dones: non-dones
            non_done_states = states[~dones]

            with torch.no_grad():
                logits = model(get_one_hot(non_done_states, h))

            prob_mask = get_mask(non_done_states, h)
            log_ProbF = torch.log_softmax(logits - 1e10 * prob_mask, -1)
            actions = log_ProbF.softmax(1).multinomial(1)

            induced_states = non_done_states + 0
            for i, action in enumerate(actions.squeeze(-1)):
                if action < n:
                    induced_states[i, action] += 1

            terminates = (actions.squeeze(-1) == n)

            # Update batches
            for s, a, t in zip(induced_states, actions, terminates.float()):
                ps, pa = get_parent_states(s, a, n)
                rs = get_rewards(s, h, R0) if t else torch.tensor(0., device=device)
                trajectories += [[ps, pa, s.view(1, -1), rs.view(-1), t.view(-1)]]

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = induced_states[~terminates]

        batch_ids = torch.LongTensor(
            (sum([[i] * len(parent_states) for i, (parent_states, _, _, _, _) in enumerate(trajectories)], []))
        ).to(device)

        parent_states, parent_actions, induced_states, rewards, finishes = [
            torch.cat(i) for i in zip(*trajectories)
        ]

        # in_flow: log F(s_{t}) = log \sum_{s' in Parent(s_{t})} exp( F(s' --> s_{t}) )
        # F(s' --> s_{t}) := F(s', a'), where s_{t} = T(s', a')
        parent_flow = model(get_one_hot(parent_states, h))
        parent_mask = get_mask(parent_states, h)
        parent_flow = (parent_flow - 1e10 * parent_mask).gather(dim=1, index=parent_actions.unsqueeze(1)).squeeze(1)
        in_flow = torch.log(
            torch.zeros((induced_states.size(0),), device=device).index_add_(0, batch_ids, torch.exp(parent_flow))
        )

        # out_flow: log F(s_{t}) = log \sum_{s'' in Children(s_{t})} exp( F(s_{t} --> s'') )
        children_flow = model(get_one_hot(induced_states, h))
        children_mask = get_mask(induced_states, h)
        children_flow = (
                (children_flow - 1e10 * children_mask) * (1 - finishes).unsqueeze(1) - 1e10 * finishes.unsqueeze(1)
        )
        out_flow = torch.logsumexp(
            torch.cat([torch.log(rewards)[:, None], children_flow], 1), -1
        )

        loss = (in_flow - out_flow).pow(2).mean()

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
