import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pickle
import glob
from collections import defaultdict
from utils import make_grid, get_rewards, get_modes_found


def get_train_args():
    parser = argparse.ArgumentParser(description='Plot results of hypergrid environment')
    parser.add_argument(
        '--n', type=int, default=2
    )
    parser.add_argument(
        '--h', type=int, default=64
    )
    parser.add_argument(
        '--R0', type=float, default=1e-2, required=True
    )
    args = parser.parse_args()
    return args


def get_stats(modes):
    all_paths = glob.glob('./*/*/out.pkl')
    all_stats = defaultdict(lambda: defaultdict(list))
    for ckpt in sorted(all_paths):
        method = ckpt.split('/')[1]
        with open(ckpt, 'rb') as f:
            if method == 'MCMC':
                total_loss = 0.
                total_reward = 0.
                total_visited_states, first_visited_states, L1 = pickle.load(f)
            else:
                total_loss, total_reward, total_visited_states, first_visited_states, L1 = pickle.load(f)
        st = [a[0] for a in L1]
        l1 = [a[1] for a in L1]
        modes_found = get_modes_found(first_visited_states, modes, num_steps=50000).cpu().numpy()
        _stats = {
            'total_loss': total_loss,
            'total_reward': total_reward,
            'total_visited_states': total_visited_states,
            'modes_found': modes_found,
            'l1': l1
        }
        for k, v in _stats.items():
            all_stats[method][k].append(v)
    aggregated_stats = defaultdict(lambda: defaultdict(list))
    for key, value in all_stats.items():
        for _key, _value in value.items():
            v = np.stack(_value)
            mv = np.mean(v, 0)
            sv = np.std(v, 0)
            aggregated_stats[key][_key] = [st, mv, sv] if _key == 'l1' else [mv, sv]
    return aggregated_stats


def main():
    args = get_train_args()

    n = args.n
    h = args.h

    grid = make_grid(n, h)
    true_rewards = get_rewards(grid, h, args.R0)
    modes = true_rewards.view(-1) >= true_rewards.max()
    num_modes = modes.sum().item()

    methods = {
        'DB_0': 'Detailed Balance, learned $P_{B}$',
        'DB_1': 'Detailed Balance, uniform $P_{B}$',
        'FM': 'Flow Matching',
        'TB_0': 'Trajectory Balance, learned $P_{B}$',
        'TB_1': 'Trajectory Balance, uniform $P_{B}$'
    }

    aggregated_stats = get_stats(modes)
    time_step = np.arange(1, 50000 + 1) * 16

    sns.set(style='darkgrid')

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['b', 'b', 'g', 'r', 'r']
    line_styles = ['b-', 'b--', 'g-', 'r-', 'r--']
    for i, (key, stat) in enumerate(aggregated_stats.items()):
        mean, std = stat['modes_found']
        ax1.plot(time_step, mean, line_styles[i], label=methods[key])
        ax1.fill_between(time_step, mean - std, mean + std, color=colors[i], alpha=0.2)
        steps, mean, std = stat['l1']
        ax2.plot(steps, mean, line_styles[i], label=methods[key])
        ax2.fill_between(steps, mean - std, mean + std, color=colors[i], alpha=0.2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    ax1.axhline(y=num_modes, color='k', linestyle='-', label='#Modes (ground-truth)')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_xlim([10**2, 10**6])
    ax1.set_xlabel('#States Visited')
    ax1.set_ylabel('#Modes Found')
    ax1.set_xscale('log')
    ax1.legend(loc=2, fontsize='medium')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_xlabel('#States Visited')
    ax2.set_ylabel('Empirical L1 Error')
    ax2.legend(loc=2, fontsize='medium')
    plt.tight_layout()
    plt.savefig('./out.png', bbox_inches='tight', format='png', dpi=200)


if __name__ == '__main__':
    main()
