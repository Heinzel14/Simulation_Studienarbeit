from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sys import argv
import matplotlib.patches as mpatches

# def drawer(netalphas, nmax, maxnetsize=None):
#     patches = [mpatches.Patch(color='blue', label='1 failure node'),
#                mpatches.Patch(color='orange', label='2 failure node'),
#                mpatches.Patch(color='green', label='3 failure node')]
#     # for netsize in range(2, maxnetsize + 1):
#     #     plt.figure(netsize - 1)
#     xmax = 0
#     for n in range(1, nmax + 1):
#         data = []
#         for key in netalphas:
#             netnow, nnow, window = key
#             if n == nnow:
#                 data += netalphas[key]
#         datasorted = sorted(data)
#         print(data)
#         print(netalphas)
#         xmax = max(xmax, max(data))
#         p = 1. * np.arange(len(datasorted)) / (len(datasorted) - 1)
#         plt.plot(datasorted, p, label='n=' + str(n), lw=3)
#     plt.xlabel('alpha',size='x-large')
#     plt.ylabel('cdf',size='x-large')
#     title = 'CDF of alpha'  # n=1 to ' + str(nmax) + ' failure nodes and a netsize of ' + str(netsize + 2) + ' nodes'
#     plt.title(title, size='x-large')
#     axes = plt.gca()
#     # if xmax > 10:
#     #   	xmax=10
#     axes.set_xlim([0, xmax])
#     axes.set_ylim([0, 1])
#     #plt.legend(loc=4)
#     # dataname = str(netsize + 2) + 'nodes'
#     # plt.savefig(dataname, bbox_inches='tight')
#     # plt.close()
#     plt.show()
#
# def volume_plotter(netalphas, SIZE, N):
#     data = []
#     y = []
#     x = []
#     z = []
#     min_len = 1000000000000
#     for key in netalphas:
#         for i in range(len(netalphas[key])):
#             if netalphas[key][i] < 0:
#                 netalphas[key][i] = 0
#
#
#     for window in range(45):
#         try:
#             data.append(sorted(netalphas[(SIZE, N, window)]))
#             if min_len > len(netalphas[(SIZE, N, window)]):
#                 min_len = len(netalphas[(SIZE, N, window)])
#
#         except:
#             print('key not found')
#             continue
#     print(min_len)
#     for i in range(len(data)):
#         data[i] = data[i][:min_len]
#
#     for n in range(len(data)):
#         try:
#             y = [*y, *((1. * np.arange(len(data[n]))) / (len(data[n]) - 1))]
#         except:
#             print(y, ((1. * np.arange(len(data[n]))) / (len(data[n]) - 1)))
#         z += data[n]
#         x += [n]*min_len
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#
#     ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
#
#     plt.show()
#
#     # # x,y,z = np.loadtxt(file, unpack=True)
#     # df = pd.DataFrame({'x': x, 'y': y, 'z': z})
#     #
#     # fig = plt.figure()
#     # ax = Axes3D(fig)
#     # surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
#     # fig.colorbar(surf, shrink=0.5, aspect=5)
#     # plt.show()

def main():
    SIZE = 14
    N = 1
    title = 'CDF for Alpha for improved filtering coefficients with sending limit'

    d2 = np.load("new_pfs_alphas.npy")
    netalphas = d2.item()
    print(min(netalphas, key=netalphas.get))
    for key in netalphas:
        if netalphas[key] < 0:
            print(key, netalphas[key])
    x = list(sorted(netalphas.values()))
    y = 1. * np.arange(len(x)) / (len(x) - 1)
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('alpha', size='x-large')
    plt.ylabel('cdf', size='x-large')
    axes = plt.gca()
    axes.set_xlim([-1, 8])
    axes.set_ylim([0, 1])
    plt.plot(x,y)
    plt.show()
    # fig.savefig(title+'.png')
    # print(netalphas)
    # volume_plotter(netalphas, SIZE, N)


if __name__ == '__main__':
    main()
