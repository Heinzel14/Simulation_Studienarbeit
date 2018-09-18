from mesh_simulator import calc_tot_send_time
from network import load_dst_coop_groups
import numpy as np


def calculate_no_feedback_alphas():

    netalphas = {}
    dstalphas = {}

    dst_coop_groups = load_dst_coop_groups()
    for dst in dst_coop_groups:
        print('calculating for dst ', dst)
        COOP_GROUPS = dst_coop_groups[dst]

        priorities = {node: COOP_GROUPS[node].priority for node in COOP_GROUPS}
        sortednodes = sorted(priorities, key=priorities.__getitem__)

        # we need at least two nodes between source and dst
        for i in range(len(sortednodes) - 2):
            source = sortednodes[i]
            m0 = calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=False, failnodes=[], source_feedback= True)
            m_res = calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=True, failnodes=[], source_feedback= True)
            alpha = (m_res/m0)-1
            if alpha < 0:
                print (source, dst, alpha)
                raise NameError('negative alpha ')

            dstalphas[(dst,source)] = alpha # ((calculate_data(source, 1, 0, 1, testmode=True) / m0) - 1)

    print('finished alpha calculation')
    return dstalphas


def main():
    np.save("old_pfs_alphas.npy", calculate_no_feedback_alphas())


if __name__ == '__main__':
    main()
