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
c = None
num_steps = 50000


def main():

    grid = make_grid(n, h)

    true_rewards = get_rewards(grid, h, R0=0.1)
    modes = true_rewards.view(-1) >= true_rewards.max() if c is None else true_rewards.view(-1) >= c
    num_modes = modes.sum().item()

    figure, axs = plt.subplots(1, 2, figsize=(15, 8))

    styles = ['b-', 'g-', 'y-', 'k-', 'c-', 'm-', 'r-', 'r--']

    for i, ckpt in enumerate(glob.glob('./*/*/out.pkl')):
        print(i)
        with open(ckpt, 'rb') as f:
            first_visited_states, L1 = pickle.load(f)[-2:]
        modes_found = get_modes_found(first_visited_states, modes, num_steps).cpu().numpy()
        ns = [a[0] for a in L1]
        l1 = [a[1] for a in L1]
        axs[0].plot(
            np.arange(1, num_steps + 1) * bsz, modes_found, styles[i], label=ckpt
        )
        axs[0].tick_params(axis='both', which='major', labelsize=10)
        axs[0].set_title(r'$%d \times %d,~R_{0} = 10^{-2}$' % (h, h), fontsize=12)
        axs[0].set_ylabel(r'#Modes Found (Maximum=$%d$)' % num_modes, fontsize=12)
        axs[0].set_xscale('log')
        axs[0].set_xlim([10 ** 2, 10 ** 6])
        axs[0].grid(linestyle='--')
        axs[0].legend()
        axs[1].plot(ns, l1, styles[i], label=ckpt)
        axs[1].tick_params(axis='both', which='major', labelsize=10)
        axs[1].set_xlabel('#States Visited', fontsize=12)
        axs[1].set_ylabel('Empirical L1 Error', fontsize=12)
        axs[1].set_xscale('log')
        axs[1].set_xlim([10 ** 3, 10 ** 6])
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        axs[1].grid(linestyle='--')

    plt.savefig('./out.png', bbox_inches='tight', format='png', dpi=300)


if __name__ == '__main__':
    main()
