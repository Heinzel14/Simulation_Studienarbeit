import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


def load_np_file(path):
    return np.load(path).item()


# group is failures and member strategy
def bar_plot(group_number, member_number, group_dict):
    group_dict
    group_member_means = {}
    group_member_stds = {}
    group_labels = []
    for member in group_dict:
        group_member_means[member] = []
        group_member_stds[member] = []

        for group in range(group_number):
            if group not in group_labels:
                group_labels.append(str(group))
            group_member_means[member].append(np.mean(group_dict[member][str(group)]))
            group_member_stds[member].append(0)#np.std(group_dict[member][str(group)]))
    fig, ax = plt.subplots()

    index = np.arange(group_number)
    print(index)
    bar_width = 0.3

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    bars = []
    colors = ['b', 'r', 'g']
    color_counter = 0
    for member in group_member_means:
        bars.append(ax.bar(index+bar_width*color_counter, tuple(group_member_means[member]), bar_width,
                    alpha=opacity, color=colors[color_counter],
                    yerr=tuple(group_member_stds[member]), error_kw=error_config,
                    label=member))
        color_counter += 1

    ax.set_xlabel('failed edges')
    ax.set_ylabel('average sending time')
    ax.set_title('Average Sending Time for Different Strategies')
    ax.set_xticks(index + bar_width/2*(member_number-1))
    ax.set_xticklabels(tuple(group_labels))
    ax.legend()

    fig.tight_layout()
    plt.show()


def time_plot(dict):
    plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
    legend = []
    for key in dict:
        legend.append(key)
        plt.plot(np.arange(len(dict[key])), dict[key], linestyle=':')
    plt.legend(legend, loc='upper left')
    plt.show()


def main():
    # dict = load_np_file("send_time_filter_rules_node_failures.npy")
    # dict = load_np_file("send_fime_filter_rules_failures.npy")
    # dict = load_np_file("send_time_filter_rules_node_failures.npy")
    # bar_plot(4, 2, dict)
    dict = load_np_file("send_time_filter_rules_over_time_no_failures.npy")
    del dict['normal']
    del dict['ff_fe']
    time_plot(dict)


if __name__ == '__main__':
    main()

