import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import glob
from collections import defaultdict
from utils import make_grid, get_rewards, get_modes_found


n = 4
h = 8
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

    figure, axs = plt.subplots(3, 3, figsize=(13, 10))

    cmap = sns.color_palette('tab10')
    colors = [
        cmap[3], cmap[3], cmap[2], cmap[6], cmap[5], cmap[1], cmap[7], cmap[0], cmap[0]
    ]
    styles = ['-', '--', '-', '-', '-', '-', '-', '-', '--']

    for i, data in enumerate([aggregated_stats_1, aggregated_stats_2, aggregated_stats_3]):
        for j, (key, stat) in enumerate(data.items()):
            if key not in methods.keys():
                continue
            else:
                mean, std = stat['modes_found']
                axs[i, 0].plot(
                    np.arange(1, num_steps + 1) * bsz, mean, c=colors[j], ls=styles[j], label=methods[key]
                )
                axs[i, 0].fill_between(
                    np.arange(1, num_steps + 1) * bsz, mean - std, mean + std, color=colors[j], alpha=0.2
                )
                axs[i, 0].tick_params(axis='both', which='major', labelsize=10)
                axs[i, 0].set_ylabel(r'#Modes Found (Maximum=$%d$)' % num_modes, fontsize=10)
                axs[i, 0].set_xscale('log')
                axs[i, 0].set_xlim([10 ** 2, 10 ** 6])
                axs[i, 0].grid(linestyle='--')
                mean, std = stat['l1']
                axs[i, 1].plot(
                    [int(t) for t in stat['num_states'][0]], mean, c=colors[j], ls=styles[j], label=methods[key]
                )
                axs[i, 1].fill_between(
                    [int(t) for t in stat['num_states'][0]], mean - std, mean + std, color=colors[j], alpha=0.2
                )
                axs[i, 1].tick_params(axis='both', which='major', labelsize=10)
                axs[i, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                axs[i, 1].set_xscale('log')
                axs[i, 1].set_xlim([10 ** 3, 10 ** 6])
                axs[i, 1].grid(linestyle='--')
                axs[i, 1].set_ylabel('Empirical L1 Error', fontsize=10)
                axs[i, 2].plot(
                    [int(t) for t in stat['num_states'][0]], mean, c=colors[j], ls=styles[j], label=methods[key]
                )
                axs[i, 2].fill_between(
                    [int(t) for t in stat['num_states'][0]], mean - std, mean + std, color=colors[j], alpha=0.2
                )
                axs[i, 2].tick_params(axis='both', which='major', labelsize=10)
                axs[i, 2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                axs[i, 2].set_xscale('log')
                axs[i, 2].set_xlim([2 * 10 ** 5, 10 ** 6])
                axs[i, 2].set_ylim([10 ** -5, 0.5 * 10 ** -4])
                axs[i, 2].grid(linestyle='--')

                axs[0, 0].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-1}$' % (h, h, h, h), fontsize=10)
                axs[1, 0].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-2}$' % (h, h, h, h), fontsize=10)
                axs[2, 0].set_title(r'$%d \times %d \times %d \times %d,~R_{0} = 10^{-3}$' % (h, h, h, h), fontsize=10)

                axs[2, 0].set_xlabel('#States Visited', fontsize=10)
                axs[2, 1].set_xlabel('#States Visited', fontsize=10)
                axs[2, 2].set_xlabel('#States Visited', fontsize=10)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    order = [labels.index(i) for i in methods.values()]
    figure.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=10, fontsize='medium'
    )
    plt.subplots_adjust(left=0.050, right=0.988, top=0.970, wspace=0.119, hspace=0.200)
    # plt.show()
    plt.savefig('./out_.png', bbox_inches='tight', format='png', dpi=300)


if __name__ == '__main__':
    main()
