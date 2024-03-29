import torch
import torch.optim as optim
import numpy as np
import argparse
import pickle
import os
from utils import add_args, set_seed, setup, make_grid, get_rewards, make_model, get_one_hot, get_mask


def get_train_args():
    parser = argparse.ArgumentParser(description='Modified FM-based GFlowNets for hypergrid environments')
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
    device = torch.device('cpu')

    parser = get_train_args()
    args = add_args(parser)

    set_seed(args.seed)

    n = args.n
    h = args.h
    R0 = args.R0
    R1 = args.R1
    R2 = args.R2
    bsz = args.bsz

    exp_name = 'modified_fm_{}_{}_{}_{}_{}_{}'.format(n, h, R0, args.lr, args.temp, args.epsilon)
    logger, exp_path = setup(exp_name, args)

    coordinate = h ** torch.arange(n, device=device)

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0, R1, R2)
    modes = true_rewards.view(-1) >= true_rewards.max()
    num_modes = modes.sum().item()

    true_rewards = true_rewards.view((h,) * n)
    true_density = true_rewards.log().flatten().softmax(0).cpu().numpy()

    first_visited_states = -1 * np.ones_like(true_density)

    model = make_model([n * h] + [args.hidden_size] * args.num_layers + [n + 2])
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

        batches = []

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
            for c, a, t in zip(child_states, actions, terminates.float()):
                ps, pa = get_parent_states(c, a, n)
                lr = get_rewards(c, h, R0, R1, R2).log() if t else torch.tensor(0., device=device)
                batches += [[ps, pa, c.view(1, -1), lr.view(-1), t.view(-1)]]

            for state in non_done_states[terminates]:
                state_id = (state * coordinate).sum().item()
                total_visited_states.append(state_id)
                if first_visited_states[state_id] < 0:
                    first_visited_states[state_id] = step

            # Update dones
            dones[~dones] |= terminates

            # Update non-done trajectories
            states[~dones] = child_states[~terminates]

        # Different number of parent states and children states (#parent_states > #child_states)
        parent_states, parent_actions, child_states, log_rewards, finishes = [
            torch.cat(i) for i in zip(*batches)
        ]

        batch_ids = torch.LongTensor(
            (sum([[i] * len(parent_states) for i, (parent_states, _, _, _, _) in enumerate(batches)], []))
        ).to(device)

        # log F(s_{t+1}) = log \sum_{s_{t} in Parent(s_{t+1})} P_F(s_{t+1} | s_{t}) F(s_{t})
        p_outputs = model(get_one_hot(parent_states, h))
        log_flowP, logits_PF = p_outputs[:, n + 1], p_outputs[:, :n + 1]
        prob_mask = get_mask(parent_states, h)
        log_ProbF = torch.log_softmax(logits_PF - 1e10 * prob_mask, -1)
        log_PF_sa = log_ProbF.gather(dim=1, index=parent_actions.unsqueeze(1)).squeeze(1)
        log_in_flow = torch.log(
            torch.zeros((child_states.size(0),), device=device).index_add_(0, batch_ids, (log_flowP + log_PF_sa).exp())
        )

        # log F(s_{t+1})
        c_outputs = model(get_one_hot(child_states, h))
        log_flowC = c_outputs[:, n + 1]
        log_flowC = log_flowC * (1 - finishes) + log_rewards * finishes

        loss = (log_in_flow - log_flowC).pow(2).mean()

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
