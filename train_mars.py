import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot


def get_train_args():
    parser = argparse.ArgumentParser(description='MARS for hypergrid environment')
    parser.add_argument(
        '--dataset_size', type=int, default=0
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

    exp_name = 'mars_{}_{}_{}'.format(n, h, R0)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0)
    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n * 2])
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

    dataset = []
    total_loss = []
    total_l1_error = []
    total_visited_states = []

    # initial state: s_0 = [0, 0]
    states = torch.zeros((bsz, n), dtype=torch.long, device=device)

    for step in range(1, args.num_steps + 1):

        optimizer.zero_grad()

        pre_rewards = get_rewards(states, h, R0)

        with torch.no_grad():
            outputs = model(get_one_hot(states, h))

        # Forward
        edge_mask = (states == h - 1).float()
        log_ProbF = torch.log_softmax(outputs[:, :n] - 1e10 * edge_mask, -1)
        actions_F = (log_ProbF / args.temp).softmax(1).multinomial(1)

        # Backward
        edge_mask = (states == 0).float()
        log_ProbB = torch.log_softmax(outputs[:, n:] - 1e10 * edge_mask, -1)
        actions_B = (log_ProbB / args.temp).softmax(1).multinomial(1)

        split = torch.rand((outputs.size(0), 1), device=device) < 0.5
        actions = actions_F * split + (n + actions_B) * (~split)

        # Backward is allowed in MCMC-based approaches
        induced_states = states + 0
        for i, action in enumerate(actions.squeeze(-1)):
            if action < n:
                induced_states[i, action] = min(induced_states[i, action] + 1, h - 1)
            if action >= n:
                induced_states[i, action - n] = max(induced_states[i, action - n] - 1, 0)

        new_rewards = get_rewards(induced_states, h, R0)

        A = new_rewards / pre_rewards
        U = torch.rand((bsz,), device=device)

        aggregates = (new_rewards > pre_rewards) + (U < 0.05)

        if aggregates.float().sum():
            for s, a in zip(states[aggregates], actions[aggregates]):
                dataset.append((s.unsqueeze(0), a.unsqueeze(0)))

        updates = (A > U)

        for state in induced_states[updates]:
            state_id = (state * coordinate).sum().item()
            total_visited_states.append(state_id)
            if first_visited_states[state_id] < 0:
                first_visited_states[state_id] = step

        # Update non-updated trajectories
        states[updates] = induced_states[updates]

        if not step % 20 and len(dataset) > args.dataset_size:
            dataset = dataset[-args.dataset_size:]

        if len(dataset) < bsz:
            continue

        idxs = np.random.randint(0, len(dataset), bsz)
        parent_states, parent_actions = [
            torch.cat(i) for i in zip(*[dataset[i] for i in idxs])
        ]
        outputs = model(get_one_hot(parent_states, h))

        log_ProbF = torch.log_softmax(outputs[:, :n] - 1e10 * (parent_states == h - 1).float(), -1).gather(
            dim=1, index=torch.minimum(parent_actions, torch.tensor(n - 1))
        )
        log_ProbB = torch.log_softmax(outputs[:, n:] - 1e10 * (parent_states == 0).float(), -1).gather(
            dim=1, index=torch.maximum(parent_actions - n, torch.tensor(0))
        )
        split = parent_actions < n
        loss = -(log_ProbF * split + log_ProbB * (~split)).mean()

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if step % 100 == 0:
            emp_density = np.bincount(total_visited_states[-200000:], minlength=len(true_density)).astype(float)
            emp_density /= emp_density.sum()
            l1 = np.abs(true_density - emp_density).mean()
            total_l1_error.append((len(total_visited_states), l1))
            logger.info('Step: %d, \tLoss: %.5f, \tL1: %.5f' % (step, np.array(total_loss[-100:]).mean(), l1))

    pickle.dump(
        [total_loss, total_visited_states, first_visited_states, total_l1_error],
        open(os.path.join(exp_path, 'out.pkl'), 'wb')
    )

    logger.info('done.')


if __name__ == '__main__':
    main()
