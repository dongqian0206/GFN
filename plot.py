import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import glob
from collections import defaultdict
from utils import make_grid, get_rewards, get_modes_found


n = 2
h = 64
bsz = 16
c = None  # 0.51
num_steps = 62500


def get_stats(modes, path):
    all_stats = defaultdict(lambda: defaultdict(list))
    for ckpt in sorted(path):
        method = ckpt.split('\\')[1]
        with open(ckpt, 'rb') as f:
            first_visited_states, L1 = pickle.load(f)[-2:]
        modes_found = get_modes_found(first_visited_states, modes, num_steps).cpu().numpy()
        ns = [a[0] for a in L1]
        l1 = [a[1] for a in L1]
        _stats = {
            'modes_found': modes_found,
            'num_states': ns,
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

    true_rewards_1 = get_rewards(grid, h, R0=0.1)
    modes_1 = true_rewards_1.view(-1) >= true_rewards_1.max() if c is None else true_rewards_1.view(-1) >= c
    num_modes = modes_1.sum().item()

    true_rewards_2 = get_rewards(grid, h, R0=0.01)
    modes_2 = true_rewards_2.view(-1) >= true_rewards_2.max() if c is None else true_rewards_2.view(-1) >= c

    true_rewards_3 = get_rewards(grid, h, R0=0.001)
    modes_3 = true_rewards_3.view(-1) >= true_rewards_3.max() if c is None else true_rewards_3.view(-1) >= c

    methods = {
        'random': 'Random',
        'mcmc': 'MCMC',
        'mars': 'MARS',
        'ppo': 'PPO',
        'sac': 'SAC',
        'fm': 'FM',
        'db_0': 'DB, learned $P_{B}$',
        'db_1': 'DB, uniform $P_{B}$',
        'tb_0': 'TB, learned $P_{B}$',
        'tb_1': 'TB, uniform $P_{B}$'
    }

    aggregated_stats_1 = get_stats(
        modes_1, path=glob.glob(f'D:/saved_models/grid_nvidia_v100/grid_{n}_{h}/grid_0.1/*/*/out.pkl')
    )
    aggregated_stats_2 = get_stats(
        modes_2, path=glob.glob(f'D:/saved_models/grid_nvidia_v100/grid_{n}_{h}/grid_0.01/*/*/out.pkl')
    )
    aggregated_stats_3 = get_stats(
        modes_3, path=glob.glob(f'D:/saved_models/grid_nvidia_v100/grid_{n}_{h}/grid_0.001/*/*/out.pkl')
    )

    figure, axs = plt.subplots(2, 3, figsize=(15, 8))

    cmap = sns.color_palette('tab10')
    colors = [
        cmap[3], cmap[3], cmap[2], cmap[6], cmap[5], cmap[1], cmap[7], cmap[8], cmap[0], cmap[0]
    ]
    styles = ['-', '--', '-', '-', '-', '-', '-', '-', '-', '--']

    for i, data in enumerate([aggregated_stats_1, aggregated_stats_2, aggregated_stats_3]):
        for j, (key, stat) in enumerate(data.items()):
            mean, std = stat['modes_found']
            axs[0, i].plot(
                np.arange(1, num_steps + 1) * bsz, mean, c=colors[j], ls=styles[j], label=methods[key]
            )
            axs[0, i].fill_between(
                np.arange(1, num_steps + 1) * bsz, mean - std, mean + std, color=colors[j], alpha=0.2
            )
            mean, std = stat['l1']
            axs[1, i].plot(
                [int(t) for t in stat['num_states'][0]], mean, c=colors[j], ls=styles[j], label=methods[key]
            )
            axs[1, i].fill_between(
                [int(t) for t in stat['num_states'][0]], mean - std, mean + std, color=colors[j], alpha=0.2
            )
            axs[0, i].tick_params(axis='both', which='major', labelsize=10)
            if i == 0:
                axs[0, i].set_title(r'$%d \times %d,~R_{0} = 10^{-1}$' % (h, h), fontsize=12)
                # axs[0, i].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-1}$' % (h, h, h, h), fontsize=12)
            elif i == 1:
                axs[0, i].set_title(r'$%d \times %d,~R_{0} = 10^{-2}$' % (h, h), fontsize=12)
                # axs[0, i].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-2}$' % (h, h, h, h), fontsize=12)
            else:
                axs[0, i].set_title(r'$%d \times %d,~R_{0} = 10^{-3}$' % (h, h), fontsize=12)
                # axs[0, i].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-3}$' % (h, h, h, h), fontsize=12)
            if i == 0:
                axs[0, i].set_ylabel(r'#Modes Found (Maximum=$%d$)' % num_modes, fontsize=12)
            axs[0, i].set_xscale('log')
            axs[0, i].set_xlim([10 ** 2, 10 ** 6])
            axs[0, i].grid(linestyle='--')
            axs[1, i].tick_params(axis='both', which='major', labelsize=10)
            axs[1, i].set_xlabel('#States Visited', fontsize=12)
            if i == 0:
                axs[1, i].set_ylabel('Empirical L1 Error', fontsize=12)
            axs[1, i].set_xscale('log')
            axs[1, i].set_xlim([1.6 * 10 ** 3, 10 ** 6])
            axs[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
            axs[1, i].grid(linestyle='--')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    order = [labels.index(i) for i in methods.values()]
    figure.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc='lower center', bbox_to_anchor=(0.5, -0.001), ncol=10, fontsize='medium'
    )
    plt.subplots_adjust(left=0.045, right=0.980, top=0.960, wspace=0.120, hspace=0.160)
    plt.show()
    # plt.savefig('./out.png', bbox_inches='tight', format='png', dpi=300)


if __name__ == '__main__':
    main()
