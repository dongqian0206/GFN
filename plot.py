import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import glob
from collections import defaultdict
from utils import make_grid, get_rewards, get_modes_found


n = 2
h = 64
R0 = 1e-2
bsz = 16
threshold = None  # 0.51
num_steps = 50000


def get_stats(modes):
    all_paths = glob.glob('D:/saved_models/grid_nvidia_v100/grid_0.01/*/*/out.pkl')
    all_stats = defaultdict(lambda: defaultdict(list))
    for ckpt in sorted(all_paths):
        method = ckpt.split('\\')[1]
        with open(ckpt, 'rb') as f:
            first_visited_states, L1 = pickle.load(f)[-2:]
        modes_found = get_modes_found(first_visited_states, modes, num_steps).cpu().numpy()
        ns = [a[0] for a in L1]
        l1 = [a[1] for a in L1]
        _stats = {
            'modes_found': modes_found,
            'num_visited_states_so_far': ns,
            'l1': l1
        }
        for k, v in _stats.items():
            all_stats[method][k].append(v)
    aggregated_stats = defaultdict(lambda: defaultdict(list))
    for method, value in all_stats.items():
        for _key, _value in value.items():
            v = np.stack(_value)
            aggregated_stats[method][_key] = [np.mean(v, 0), np.std(v, 0)]
    return aggregated_stats


def main():

    grid = make_grid(n, h)
    true_rewards = get_rewards(grid, h, R0)
    modes = true_rewards.view(-1) >= true_rewards.max() if threshold is None else true_rewards.view(-1) >= threshold
    num_modes = modes.sum().item()

    plt.figure()
    sns.heatmap(true_rewards.view((h,) * n).cpu().numpy(), cmap='Blues', linewidths=0.5, square=True, vmin=0., vmax=3.)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.title(r'$D = %d,~H = %d,~R_{0} = 10^{-2}$' % (n, h))
    plt.tight_layout()
    plt.savefig('./grid.png', bbox_inches='tight', format='png', dpi=300)

    methods = {
        'random': 'Random',
        'mcmc': 'MCMC',
        # 'mars': 'MARS',
        # 'ppo': 'PPO',
        'sac': 'SAC',
        'fm': 'FM',
        'db_0': 'DB, learned $P_{B}$',
        'db_1': 'DB, uniform $P_{B}$',
        'tb_0': 'TB, learned $P_{B}$',
        'tb_1': 'TB, uniform $P_{B}$'
    }

    aggregated_stats = get_stats(modes)

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['b', 'b', 'g', 'y', 'k', 'c', 'r', 'r']
    styles = ['b-', 'b--', 'g-', 'y-', 'k-', 'c-', 'r-', 'r--']
    for i, (key, stat) in enumerate(aggregated_stats.items()):
        mean, std = stat['modes_found']
        ax1.plot(
            np.arange(1, num_steps + 1) * bsz, mean, styles[i], label=methods[key]
        )
        ax1.fill_between(
            np.arange(1, num_steps + 1) * bsz, mean - std, mean + std, color=colors[i], alpha=0.2
        )
        mean, std = stat['l1']
        ax2.plot(
            [int(t) for t in stat['num_visited_states_so_far'][0]], mean, styles[i], label=methods[key]
        )
        ax2.fill_between(
            [int(t) for t in stat['num_visited_states_so_far'][0]], mean - std, mean + std, color=colors[i], alpha=0.2
        )
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title(r'$%d \times %d,~R_{0} = 10^{-2}$' % (h, h))
    ax1.set_xlabel('#States Visited')
    ax1.set_ylabel(r'#Modes Found (Maximum=$%d$)' % num_modes)
    ax1.set_xscale('log')
    ax1.set_xlim([10 ** 2, 10 ** 6])
    ax1.grid(linestyle='--')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [labels.index(i) for i in methods.values()]
    ax1.legend([handles[i] for i in order], [labels[i] for i in order], loc=2, fontsize='medium')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(r'$%d \times %d,~R_{0} = 10^{-2}$' % (h, h))
    ax2.set_xlabel('#States Visited')
    ax2.set_ylabel('Empirical L1 Error')
    ax2.set_xscale('log')
    ax2.set_xlim([10 ** 3, 10 ** 6])
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    ax2.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('./out.png', bbox_inches='tight', format='png', dpi=300)


if __name__ == '__main__':
    main()
