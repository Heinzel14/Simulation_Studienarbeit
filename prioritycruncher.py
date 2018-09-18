import json
import numpy as np
from functools import reduce
import operator
import os
from itertools import tee
from global_variables import MAX_LOSS, WINDOW_SIZE
#import dijkstra

mcs_dr_20Mhz_GI = {0: 6.5,
                   1: 13,
                   2: 19.5,
                   3: 26,
                   4: 39,
                   5: 52,
                   6: 58.5,
                   7: 65,
                   8: 13,
                   9: 26,
                   10: 39,
                   11: 52,
                   12: 72,
                   13: 104,
                   14: 117,
                   }

def partition(alist, indice):

    indicelist = []
    for i in range(int(len(alist)/indice)-1):
        indicelist.append(indice*(i+1))
    return [alist[i:j] for i, j in zip([0]+indicelist, indicelist+[None])]

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Network:
    def __init__(self, data_root=None, crunched_root=None):
        self.links = []
        self.nodes = {}
        self.bitmask_len = None
        self.data_root = data_root
        self.max_rounds = {}
        self.max_error = MAX_LOSS
        self.window_size = WINDOW_SIZE
        self.window_counter = -1


        # Dijkstra calculation
        self.simplified_edges = {}
        self.simplified_distances = {}

        if data_root:
            self.get_max_rounds()
            self.gen_links(data_root)
            self.gen_simplified_edges()
            self.init_priorities()
            self.size_windows(WINDOW_SIZE)
        elif crunched_root:
            self.load_net(crunched_root)
        else:
            exit('Provide path to the log dump')

    def init_priorities(self):
        for node in self.nodes:
            for dst_node in self.nodes:
                if node == dst_node:
                    self.set_priority(node, dst_node, float('inf'))
                else:
                    self.set_priority(node, dst_node, 0)

    def dump_net(self, outfile='net_dump.npy'):
        net = {'nodes': self.nodes,
               'links': self.links,
               'bitmask_len': self.bitmask_len,
               'max_rounds': self.max_rounds,
               'simplified_edges': self.simplified_edges,
               'simplified_distances': self.simplified_distances}
        np.save(outfile, net)

    # noinspection PyUnresolvedReferences
    def load_net(self, dumpfile='net_dump.npy'):
        net = np.load(dumpfile)
        self.nodes = net.item().get('nodes')
        self.links = net.item().get('links')
        self.bitmask_len = net.item().get('bitmask_len')
        self.max_rounds = net.item().get('max_rounds')
        self.simplified_edges = net.item().get('simplified_edges')
        self.simplified_distances = net.item().get('simplified_distances')
        # initialize windows
        self.size_windows(WINDOW_SIZE)
    def get_labels(self, label_file):
        lables = {}
        if os.path.isfile(label_file):
            with open(label_file) as f:
                for line in f:
                    kv_pair = line.split(" ")
                    lables[kv_pair[0]] = kv_pair[1].rstrip('\n')
        else:
            lables = {node: node for node in os.listdir(self.data_root) if
                      os.path.isdir(os.path.join(self.data_root, node))}
            print(lables)

        return lables

    def get_max_rounds(self):
        labels = self.get_labels(os.path.join(self.data_root, 'labels.txt'))
        nodes = [node for node in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, node))]

        for node in nodes:
            for rate_dump in os.listdir(os.path.join(self.data_root, node)):
                # hidden .directory file in dolphin is causing trouble
                if str(rate_dump) == '.directory':
                    continue

                src_mac = rate_dump[14:31]

                with open(os.path.join(self.data_root, node, rate_dump)) as dump:
                    print(dump)
                    data = json.load(dump)
                    src = labels[src_mac]
                    if src not in self.max_rounds:
                        self.max_rounds[src] = data[-1]['round']
                    else:
                        self.max_rounds[src] = max(self.max_rounds[src], data[-1]['round'])

        # Decrement max_round count by 1 to ignore partially sent rounds
        for node in self.max_rounds.keys():
            self.max_rounds[node] -= 1

    def calc_optimal_mcs(self, mcs_bitmasks, src):
        effective_dr = {}
        bitmap_len = self.bitmask_len * self.max_rounds[src]
        for mcs in mcs_dr_20Mhz_GI.keys():
            effective_dr[mcs] = mcs_dr_20Mhz_GI[mcs] * \
                                (np.count_nonzero(mcs_bitmasks[str(mcs)]['bitmap'])) / bitmap_len
        optimal_mcs, optimal_dr = max(effective_dr.items(), key=operator.itemgetter(1))
        return optimal_mcs, optimal_dr

    def get_optimal_link_mcs(self, src, dst):
        for link in self.links:
            if link['src'] == src and link['dst'] == dst:
                optimal_mcs = link['optimal_mcs']
                return optimal_mcs

    def crunch_loss_bitmap(self, filepath, src, dst):
        with open(filepath) as dump:
            data = json.load(dump)

        mcs_bitmasks = {}
        mcs_empty_bitmasks = {}
        for mcs in range(15):
            bitmap = ''
            if not self.bitmask_len:
                self.bitmask_len = len(data[0]['bitmask'])

            max_rounds = data[-1]['round']

            # We skip the first and last rounds
            for round_i in range(1, max_rounds):
                for row in data:
                    if row.get('round') == round_i and row.get('mcs') == mcs:
                        bitmap += row['bitmask']
                        break
                    elif row is data[-1]:
                        # If the round does not have a mcs bitmask append zeroes
                        bitmap += '0' * self.bitmask_len

            mcs_bitmasks[str(mcs)] = {"datarate": mcs_dr_20Mhz_GI[mcs],
                                      #  Represent as numpy int (bit) array for efficiency during OR op
                                      "bitmap": np.fromstring(bitmap, 'u1') - ord('0')
                                      }

            mcs_empty_bitmasks[str(mcs)] = {}




        optimal_mcs, optimal_dr = self.calc_optimal_mcs(mcs_bitmasks, src)

        crunched = {'src': src,
                    'dst': dst,
                    'optimal_mcs': optimal_mcs,
                    'optimal_dr': optimal_dr,
                    'mcs_bitmasks': mcs_bitmasks,
                    'err_cache': {},
                    # also contains the bitmasks but cut into windows of the same size
                    'mcs_bitmask_windows': mcs_empty_bitmasks
                    }
        return crunched

    def gen_links(self, data_root):
        labels = self.get_labels(os.path.join(data_root, 'labels.txt'))
        for label in labels.values():
            self.nodes[label] = {'priority': {},
                                 'redundancy': {},
                                 'ideal_mcs_to_dst': {},
                                 'max_node_priority': {}
                                 }

        nodes = [node for node in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, node))]

        for node in nodes:
            for rate_dump in os.listdir(os.path.join(data_root, node)):

                # hidden .directory file in dolphin is causing trouble
                if str(rate_dump) == '.directory':
                    continue

                dst = node
                src = rate_dump[14:31]

                self.links.append(self.crunch_loss_bitmap(os.path.join(data_root, node, rate_dump),
                                                          src=labels[src],
                                                          dst=labels[dst]))
        for link in self.links:
            bitmap_len = self.bitmask_len * self.max_rounds[link['src']]
            for mcs_bitmask in link['mcs_bitmasks']:
                zeroes_to_append = bitmap_len - len(link['mcs_bitmasks'][mcs_bitmask]['bitmap'])
                if zeroes_to_append:
                    link['mcs_bitmasks'][mcs_bitmask]['bitmap'] = \
                        np.concatenate((link['mcs_bitmasks'][mcs_bitmask]['bitmap'], np.zeros(zeroes_to_append)))


    def gen_simplified_edges(self):
        for node in self.nodes.keys():
            self.simplified_edges[node] = \
                {link['dst']: link['optimal_dr'] for link in self.links if link['src'] == node}
            self.simplified_distances[node] = \
                {link['dst']: 1 / link['optimal_dr'] if link['optimal_dr'] else float('inf')
                 for link in self.links if link['src'] == node}

    def ideal_coop_mcs(self, node, dst, max_neigh, handle_correlation):
        mcs_priority_list = {mcs: self.compute_node_priority(node, dst, mcs, max_neigh, handle_correlation)
                             for mcs in range(15)}
        ideal_coop_mcs = max(mcs_priority_list, key=mcs_priority_list.get)
        return ideal_coop_mcs

    def get_priority(self, node, dst, max_neigh=None):
        if max_neigh:
            return self.get_max_node_priority(node, dst, max_neigh)
        else:
            return self.nodes[node]['priority'][dst]

    def set_priority(self, node, dst, priority):
        self.nodes[node]['priority'][dst] = priority

    def get_max_node_priority(self, node, dst, max_nodes):
        if dst not in self.nodes[node]['max_node_priority'][max_nodes]:
            return float('inf')
        return self.nodes[node]['max_node_priority'][max_nodes][dst]

    def set_max_node_priority(self, node, dst, max_nodes, priority):
        if 'max_node_priority' not in self.nodes[node]:
            self.nodes[node]['max_node_priority'] = {}
        if max_nodes not in self.nodes[node]['max_node_priority']:
            self.nodes[node]['max_node_priority'][max_nodes] = {}
        self.nodes[node]['max_node_priority'][max_nodes][dst] = priority

    def get_bitmap(self, src, dst, mcs):
        for link in self.links:
            if link['src'] == src and link['dst'] == dst:
                return link['mcs_bitmasks'][str(mcs)]['bitmap']

    def get_err(self, src, dst, mcs):
        lnk = self.get_link(src, dst)
        return self.get_link_err(lnk, mcs) if lnk else 1.0

    def get_link_err(self, link, mcs):
        if 'err_cache' not in link:
            link['err_cache'] = {}
        if mcs not in link['err_cache']:
            if self.window_counter == -1:
                bitmap_len = self.bitmask_len * self.max_rounds[link['src']]
            # if we are windowing the size of the bitmap is the window size
            else:
                bitmap_len = self.window_size
            link['err_cache'][mcs] = \
                (bitmap_len - np.count_nonzero(link['mcs_bitmasks'][str(mcs)]['bitmap'])) / bitmap_len \
                    if str(mcs) in link['mcs_bitmasks'] else 1

        # raise error if problem with window error rates
        if link['err_cache'][mcs] < 0 or link['err_cache'][mcs] > 1:
            print(link['err_cache'][mcs], bitmap_len, len(link['mcs_bitmasks'][str(mcs)]['bitmap']))
            raise NameError('not possible error rate')
        return link['err_cache'][mcs]

    def get_link(self, src, dst):
        for link in self.links:
            if link['src'] == src and link['dst'] == dst:
                return link

    # def unipath_datarate(self, src, dst):
    #     route, _ = dijkstra.shortest_path(self.simplified_distances, src, dst)
    #     min_dr = 1 / sum([self.simplified_distances[s][d] for s, d in pairwise(route)])
    #     return min_dr
    #
    # def get_unipath_nexthop_mcs(self, src, dst):
    #     route, _ = dijkstra.shortest_path(self.simplified_distances, src, dst)
    #     first_hop = route[1]
    #     return self.get_optimal_link_mcs(src, first_hop)

    def get_audible_neighbours(self, threshold=0.9):
        num_audible = []
        for src in self.nodes:
            for dst in self.nodes:
                if not src == dst:
                    audible = 0
                    for neigh in self.nodes:
                        #We are chosing a specific MCS rate here.
                        if self.get_err(src, neigh, 0) < threshold:
                            audible += 1
                    num_audible.append(audible)

        return num_audible

    # noinspection PyArgumentList
    @staticmethod
    def mutual_info(links, mcs):
        link_bitmaps = [link['mcs_bitmasks'][str(mcs)]['bitmap'] for link in links]
        return np.logical_or.reduce(link_bitmaps)

    def mutual_error(self, links, mcs):
        if len(links) == 0:
            return 1
        src = links[0]['src']
        for link in links:
            # All the links must have the same src to calculate the mutual errors
            assert src == link['src']

        bitmap_len = self.bitmask_len * self.max_rounds[src]
        return (bitmap_len - np.count_nonzero(self.mutual_info(links, mcs))) / bitmap_len

    def egress_links(self, node, dst, max_links, mcs):
        egress_links = []
        # for link in self.links:
        #     # looks for links to neighbours with errors lower than the max error and higher priority
        #     if link['src'] == node and self.get_priority(node, dst) < self.get_priority(link['dst'], dst)\
        #             and self.get_err(node, link['dst'], mcs) < self.max_error:
        #         egress_links.append(link)

        if max_links:
            for link in self.links:
                # looks for links to neighbours with errors lower than the max error and higher priority
                if link['src'] == node and self.get_max_node_priority(node, dst, max_links) < self.get_max_node_priority(link['dst'], dst, max_links) \
                        and self.get_err(node, link['dst'], mcs) < self.max_error:
                    egress_links.append(link)

            sorted_links = sorted(egress_links,
                                  # key=lambda lnk: self.airtime_of_neigh_to_dst
                                  # (node, lnk['dst'], dst, max_links, handle_correlation=False, mcs=mcs),
                                  key=lambda lnk: self.get_max_node_priority(lnk['dst'], dst, max_links),
                                  # key=lambda lnk: lnk['optimal_dr'],
                                  reverse=True)
            return sorted_links[:max_links]
        else:
            for link in self.links:
                # looks for links to neighbours with errors lower than the max error and higher priority
                if link['src'] == node and self.get_priority(node, dst) < self.get_priority(link['dst'], dst)\
                        and self.get_err(node, link['dst'], mcs) < self.max_error:
                    egress_links.append(link)
            return egress_links

    def egress_neigh(self, node, dst, max_neigh, mcs):
        return [link['dst'] for link in self.egress_links(node, dst, max_neigh, mcs)]

    def superior_egress_links(self, node, neigh, dst, max_links, mcs, include_self):
        egress_links = self.egress_links(node, dst, max_links, mcs)
        superior_egress_links = \
            [link for link in egress_links if self.get_priority(link['dst'], dst) > self.get_priority(neigh, dst)]
        if include_self:
            superior_egress_links.append(self.get_link(node, neigh))
        return superior_egress_links

    def zhenya_coeff(self, node, neigh, dst, max_neigh, handle_correlation=False, mcs=None):
        if mcs is None:
            mcs = self.ideal_coop_mcs(node, dst, max_neigh=max_neigh, handle_correlation=handle_correlation)
        p1 = 1 / self.get_max_node_priority(neigh, dst, max_neigh)
        err = self.get_err(node, neigh, mcs)
        if err == 1:
            return 0

        z = 1 / (p1 + 1 / (mcs_dr_20Mhz_GI[mcs] * (1 - err)))
        return z

    def airtime_of_neigh_to_dst(self, node, neigh, dst, max_neigh, handle_correlation=False, mcs=None):
        if mcs is None:
            mcs = self.ideal_coop_mcs(node, dst, max_neigh=max_neigh, handle_correlation=handle_correlation)
        b = self.b(node, neigh, dst,
                   mcs=mcs,
                   max_neigh=None,
                   handle_correlation=handle_correlation
                   )
        p = self.get_priority(neigh, dst)
        airtime_used = b / p
        return airtime_used

    def get_gain(self, src, dst, max_grp_size=None):
        unipath_rate = self.unipath_datarate(src, dst)
        p = self.get_max_node_priority(src, dst, max_grp_size) if max_grp_size else self.get_priority(src, dst)
        gain = p / unipath_rate
        # print(p, unipath_rate, max_grp_size, src, dst, gain)
        return gain

    def a(self, node, dst, mcs, max_neigh, handle_correlation=False):
        if not handle_correlation:
            egress_errors = [self.get_link_err(link, mcs) for link in self.egress_links(node, dst, max_neigh, mcs)]
            # if there is no neighbour a should be 0
            if len(egress_errors) == 0:
                return 0
            else:
                a = mcs_dr_20Mhz_GI[mcs] * (1 - prod(egress_errors))
                return a
        else:
            mutual_error = self.mutual_error(self.egress_links(node, dst, max_neigh, mcs), mcs)
            a = mcs_dr_20Mhz_GI[mcs] * (1 - mutual_error)
            return a

    def b(self, node, neigh, dst, mcs, max_neigh, handle_correlation=False):
        from coefficients_getter import get_greedy_stategy_pfs
        if not handle_correlation:
            superior_egress_errors = \
                [self.get_link_err(link, mcs) for link in
                 self.superior_egress_links(node, neigh, dst, max_neigh, mcs,include_self=False)]

            # # for fair pfs a dict of the losses of all neighbours is needed:
            # egress_error_dict = \
            #     {link['dst']: self.get_link_err(link, mcs) for link in
            #      self.egress_links(node, dst, max_neigh, mcs)}
            # priority_dict = {link['dst']: self.get_priority(node, dst, max_neigh) for link in
            #                  self.egress_links(node, dst, max_neigh, mcs)}

            # pf_list, fair_pf_dict, c = get_greedy_stategy_pfs(priority_dict, egress_error_dict)

            # here is the forwarding coefficient, change here to switch forwarding strategy
            pf = prod(superior_egress_errors) #fair_pf_dict[neigh]#
            b = mcs_dr_20Mhz_GI[mcs] * (1 - self.get_err(node, neigh, mcs)) * pf
            return b
        else:
            l1 = self.mutual_error(self.superior_egress_links(node, neigh, dst, max_neigh, mcs=mcs, include_self=False),
                                   mcs)
            l2 = self.mutual_error(self.superior_egress_links(node, neigh, dst, max_neigh, mcs=mcs, include_self=True),
                                   mcs)

            b = mcs_dr_20Mhz_GI[mcs] * (l1 - l2)
            return b

    def compute_node_priority(self, node, dst, mcs, max_neigh, handle_correlation=False, verbose=False, fair=False):
        a = self.a(node, dst,
                   mcs=mcs,
                   max_neigh=max_neigh,
                   handle_correlation=handle_correlation)
        if verbose:
            print('\nNode: {}  MCS: {} \nA = {:0.2f}'.format(node, mcs, a))
        b_ovr_p_list = []
        for neigh in self.egress_neigh(node, dst, max_neigh, mcs):
            if not neigh == dst:
                b = self.b(node, neigh, dst,
                           mcs=mcs,
                           max_neigh=max_neigh,
                           handle_correlation=handle_correlation)
                p_neigh = \
                    self.get_max_node_priority(neigh, dst, max_neigh) if max_neigh else self.get_priority(neigh, dst)
                b_ovr_p_list.append(b / p_neigh)
                if verbose:
                    print('B: {} --> {} \t\t {:0.2f}'.format(node, neigh, b))

        p = a / (1 + sum(b_ovr_p_list))
        if verbose:
            print('p = {:0.2f}'.format(p))
        return p

    def compute_priority(self, dst, max_rounds=10, max_neigh=None, handle_correlation=False):
        for i in range(max_rounds):
            print('\n{} Round {} {}'.format('*' * 30, i, '*' * 30))
            for node in self.nodes:
                if not node == dst:
                    mcs = self.ideal_coop_mcs(node, dst, max_neigh, handle_correlation)
                    p = self.compute_node_priority(node=node,
                                                   dst=dst,
                                                   mcs=mcs,
                                                   max_neigh=max_neigh,
                                                   handle_correlation=handle_correlation,
                                                   verbose=False)
                    if max_neigh:
                        self.set_max_node_priority(node, dst, max_neigh, p)
                    else:
                        self.set_priority(node, dst, p)
                    priorities = [self.get_priority(neigh, dst) for neigh in self.egress_neigh(node, dst, max_neigh, mcs)]
                    #priorities = [self.get_max_node_priority(neigh, dst, max_neigh) for neigh in self.egress_neigh(node, dst, max_neigh, mcs)]
                    # print('\nNode: {}\t\tPriority: {:6.4f}\tMCS: {}\nEgressNeigh: {}\nNeighPriority: {}'
                    #       .format(node, self.get_max_node_priority(node, dst, max_neigh),
                    #               self.ideal_coop_mcs(node, dst, max_neigh, handle_correlation),
                    #               self.egress_neigh(node, dst, max_neigh, mcs),
                    #               ['{:6.2f}'.format(p) for p in priorities]
                    #               ))
                    print('\nNode: {}\t\tPriority: {:6.4f}\tMCS: {}\nEgressNeigh: {}\nNeighPriority: {}'
                          .format(node, self.get_priority(node, dst),
                                  self.ideal_coop_mcs(node, dst, max_neigh, handle_correlation),
                                  self.egress_neigh(node, dst, max_neigh, mcs),
                                  ['{:6.2f}'.format(p) for p in priorities]
                                  ))
    #
    # def gen_dot_file_link_datarate(self, outfile='/tmp/map_DR.pdf'):
    #     import pygraphviz as pgv
    #     import random
    #
    #     colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta', 'brown', 'black']
    #     g = pgv.AGraph(directed=True, strict=True)
    #     for src in self.simplified_edges.keys():
    #         for dst in self.simplified_edges[src].keys():
    #             if self.simplified_edges[src][dst]:
    #                 label = '{}: {:https://cn.ifn.et.tu-dresden.de/contact/building/.2f}Mbps'.format(self.get_optimal_link_mcs(src, dst),
    #                                                 self.simplified_edges[src][dst])
    #                 color = random.choice(colors)
    #                 g.add_edge(src, dst,
    #                            label=label,
    #                            color=color,
    #                            fontcolor=color
    #                            )
    #     g.layout('dot')
    #     g.draw(outfile, format='pdf')
    #
    # def gen_dot_file_link_airtime(self, outfile='/tmp/map_AT.pdf'):
    #     import pygraphviz as pgv
    #     import random
    #
    #     colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta', 'brown', 'black']
    #     g = pgv.AGraph(directed=True, strict=True)
    #     for src in self.simplified_distances.keys():
    #         for dst in self.simplified_distances[src].keys():
    #             if self.simplified_edges[src][dst]:
    #                 color = random.choice(colors)
    #                 g.add_edge(src, dst,
    #                            label='{:.4f}spmb'.format(self.simplified_distances[src][dst]),
    #                            color=color,
    #                            fontcolor=color
    #                            )
    #     g.layout('dot')
    #     g.draw(outfile, format='pdf')

    def gen_vis_js_datarate(self, outfile='visualisation/netvis.json'):
        netvis = {'nodes': [{'id': node, 'label': node} for node in self.nodes.keys()],
                  'edges': []}
        for src in self.nodes.keys():
            for dst in self.nodes.keys():
                if not src == dst:
                    try:
                        direct_link_qual = self.simplified_edges[src][dst]
                    except KeyError:
                        direct_link_qual = 0

                    unipath = self.unipath_datarate(src, dst)
                    priority = self.nodes[src]['priority'][dst]
                    gain = self.nodes[src]['priority'][dst] / self.unipath_datarate(src, dst)

                    netvis['edges'].append({'from': src, 'to': dst,
                                            'label': '{:.2f}'.format(direct_link_qual),
                                            'link_qual': '{:.2f}'.format(direct_link_qual),
                                            'unipath': '{:.2f}'.format(unipath),
                                            'priority': '{:.2f}'.format(priority),
                                            'gain': '{:.2f}'.format(gain)
                                            })

        with open(outfile, 'w') as outfile:
            json.dump(netvis, outfile, indent=4, separators=(',', ': '))

    def gen_vis_js_priorities(self, outfile='visualisation/priorities.json'):
        nodes_dump = self.nodes
        for node in nodes_dump:
            for dst in nodes_dump[node]['priority']:
                if nodes_dump[node]['priority'][dst] == float('inf'):
                    nodes_dump[node]['priority'][dst] = "infinity"

        with open(outfile, 'w') as outfile:
            json.dump(nodes_dump, outfile, indent=4, separators=(',', ': '))

    # sets bitmaps to current window bitmap and deletes err_cache so that they will be calculated for current window
    def set_next_window(self):
        self.window_counter += 1
        for link in self.links:
            link['err_cache'] = {}
            for mcs in range(15):
                mcs = str(mcs)
                link['mcs_bitmasks'][mcs]['bitmap'] = \
                    link['mcs_bitmask_windows'][mcs]['bitmap'][self.window_counter]
            link['optimal_mcs'], link['optimal_dr'] = \
                self.calc_optimal_mcs(link['mcs_bitmasks'], link['src'])

    # used for initializing the windows in the first place. Can not be used after set_next_window!!!
    def size_windows(self, size):
        for link in self.links:
            for mcs in range(15):
                mcs = str(mcs)
                partitioned_bitmap = partition(link['mcs_bitmasks'][mcs]['bitmap'], size)
                link['mcs_bitmask_windows'][mcs]['bitmap'] = \
                    [partitioned_bitmap[i] for i in range(len(partitioned_bitmap))]




def main():
    # net = Network('test_data/2017-04-28-15-34')
    #net = Network(data_root='test_data/2017-7-31-11-35-46')
    net = Network(crunched_root='net_dump_priority_test.npy')

    # for dst in ['fl-print']:
    #     dst = 'fl-print'
    #     node = 'showroom'
    #     net.compute_priority(dst=dst, max_rounds=10, max_neigh=None, handle_correlation=False)
    #     mcs = net.ideal_coop_mcs(node, dst, None, False)
    #     print(net.egress_neigh(node, dst, None, mcs))

    for dst in net.nodes.keys():
         net.compute_priority(dst=dst, max_rounds=5, max_neigh=None, handle_correlation=False)

    # dst = 'frawi'
    # net.compute_priority(dst=dst, max_rounds=10, max_neigh=None, handle_correlation=False)
    # print('\n')
    # for src in net.nodes:
    #     if not src == dst:
    #         print('Shortest path: {:<35}'.format(str(dijkstra.shortest_path(net.simplified_distances, src, dst)[0])),
    #               '\tDatarate: {:8.4f}'.format(net.unipath_datarate(src, dst)))

    # net.dump_net('3_net_dump_priority.npy')
    # net.dump_net('net_dump_priority_test.npy')

    # net.gen_dot_file_link_datarate()
    # net.gen_dot_file_link_airtime()
    # net.gen_vis_js_datarate()
    # net.gen_vis_js_priorities()


if __name__ == '__main__':
    main()