import prioritycruncher as pc
import numpy as np
from global_variables import *
additional_sending_data_dict = {'testmode':{}, 'no testmode':{}}
from collections import deque
from itertools import permutations


def combinations(objects, k):
    object = list(objects)
    if objects == [] or len(objects) < k or k == 0:
        yield []
    elif len(objects) == k:
        yield objects
    else:
        for combination in combinations(objects[1:], k - 1):
            yield [objects[0]] + combination
        for combination in combinations(objects[1:], k):
            yield combination


def partition(alist, indice):

    indicelist = []
    for i in range(int(len(alist)/indice)-1):
        indicelist.append(indice*(i+1))
    return [alist[i:j] for i, j in zip([0]+indicelist, indicelist+[None])]


class Link:

    def __init__(self, source, dst):
        self.failed = False
        self.source = source
        self.dst = dst
        self.participating_paths = []

    def get_err(self, net, source, dst, mcs):
        if self.failed:
            return 1
        return net.get_err(source, dst, mcs)

    def add_participating_path(self, source, dst):
        if (source, dst) not in self.participating_paths:
            self.participating_paths.append((source, dst))

    def set_failed(self):
        self.failed = True

    def unset_failed(self):
        self.failed = False

    def reset_path(self):
        self.participating_paths = []

    def get_participating_paths(self):
        return self.participating_paths


class Vertex:

    def __init__(self, priority, name, mcs, priority_dict={}, loss_dict={}):
        self.name = name
        self.priority = priority
        self.priority_dict = priority_dict
        self.loss_dict = loss_dict
        self.datarate = pc.mcs_dr_20Mhz_GI[mcs]
        self.mcs = mcs
        self.coder = None
        self.full_rank = False
        self.sending_rank = 0
        self.packet_counter = 0
        self.total_data_sent = 0
        self.total_data_received = 0

    def add_neighbour(self, name, priority, loss):
        global WINDOW_SIZE
        self.priority_dict[name] = priority
        self.loss_dict[name] = loss

    def rm_neighbour(self, name):
        del self.priority_dict[name]
        del self.loss_dict[name]

    def get_losses(self):
        return self.loss_dict

    def get_priorities(self):
        return self.priority_dict

    def get_self_priority(self):
        return self.priority

    def get_datarate(self):
        return self.datarate

    def get_mcs(self):
        return self.mcs

    def reset_counter(self):
        self.coder = None
        self.sending_rank = 0
        self.packet_counter = 0
        self.total_data_sent = 0
        self.total_data_received = 0
        self.full_rank = False





class Network:
    def __init__(self, fair=False):

        """

        :param fair:
        :var self.links : dict of link objects
        """
        self.net = pc.Network(crunched_root='net_dump_priority_test.npy')
        self.dst_coop_groups = {}
        self.links = {}
        self.update_coop_groups(initialising=True)
        self.set_up_links()


    def set_up_links(self):
        self.links = {}
        for link in self.net.links:
            if (link['src'], link['dst']) not in self.links.keys():
                self.links[(link['src'], link['dst'])] = Link(link['src'],  link['dst'])
        for permutation in permutations(list(self.dst_coop_groups.keys()), 2):
            source, dst = permutation
            self.way_to_dst(source, dst, mark_links=True)

    def set_link_failure(self, link):
        """

        :param link: a tuple (source, dst)
        :return:
        """
        self.links[link].set_failed()

    def unset_link_failure(self, link):
        self.links[link].unset_failed()

    def set_next_loss_window(self):
        self.net.set_next_window()
        self.reset_vertex_counter()

    def reset_vertex_counter(self):
        """
        Resets vertex counter.
        :return:
        """
        for dst in self.dst_coop_groups:
            for node in self.dst_coop_groups[dst].values():
                node.reset_counter()

    def reset_failures(self):
        """
        Resets all failed edges/nodes.
        :return:
        """
        for link in self.links.values():
            link.unset_failed()

    def update_link_paths(self):
        """
        Updates the paths the links are participating in. Needed if priorities change because this can change
        the paths
        :return:
        """
        for link in self.links.values():
            link.reset_path()
        for permutation in permutations(list(self.dst_coop_groups.keys()), 2):
            source, dst = permutation
            self.way_to_dst(source, dst, mark_links=True)

    def get_link_loss(self, link_source, link_dst, mcs):
        return self.links[(link_source, link_dst)].get_err(self.net, link_source, link_dst, mcs)

    def get_ideal_link_loss(self, link_source, link_dst, total_dst):
        mcs = self.dst_coop_groups[total_dst][link_source].get_mcs()
        return self.links[(link_source, link_dst)].get_err(self.net, link_source, link_dst, mcs)

    def set_node_failure(self, fail_node):
        """
        To simulate a node failure all edges that contain this node are marked as failed
        :param fail_node: name of the node that failes
        """
        for (source, dst) in self.links:
            if source == fail_node or dst == fail_node:
                link = (source, dst)
                self.set_link_failure(link)

    def unset_node_failure(self, fail_node):
        """
        To simulate a node failure all edges that contain this node are marked as failed.
        In this function they are marked as unfailed again
        :param fail_node: name of the node that should set to not failed again
        """
        for (source, dst) in self.links:
            if source == fail_node or dst == fail_node:
                self.unset_link_failure(source, dst)

    def get_dst_coop_groups(self, dst):
        """
        Returns the cooperation groups for the given destination.
        :param dst: name of the destination node
        :return: dict of coop groups
        """

        return self.dst_coop_groups[dst]

    def way_to_dst(self, source, dst, mark_links=False):
        """
        Returns False if source lost connection to dst and True if not
        :param source:
        :param dst:
        :param mark_links: if True links on the path will be marked to it
        :return: Boolean
        """
        buffer = deque([])
        coop_group = self.dst_coop_groups[dst]
        buffer.append(coop_group[source])
        true_flag = False
        while len(buffer) > 0:
            node = buffer.popleft()
            if node.name == dst:
                true_flag = True
                if not mark_links:
                    return True
            for neigh in node.get_priorities().keys():
                if self.get_link_loss(node.name, neigh, node.mcs) < 1:
                    if mark_links:
                        self.links[node.name, neigh].add_participating_path(source, dst)
                    buffer.append(coop_group[neigh])
        return true_flag

    # needs to be adapted if using max neigh
    def update_coop_groups(self, initialising=False):
        self.dst_coop_groups = {}

        if not initialising:
            print('updating priorities')
            # failed edges have to be reseted manually if wished
            self.net.init_priorities()

        for dst in self.net.nodes.keys():
            # for initializing the priorities do not need to be updated
            if not initialising:
                self.net.compute_priority(dst, max_rounds=10, max_neigh=MAX_NEIGH
                                          , handle_correlation=False)
            coop_group = {}
            for node in self.net.nodes.keys():
                mcs = self.net.ideal_coop_mcs(node, dst, max_neigh=MAX_NEIGH, handle_correlation=False)
                coop_group[node] = Vertex(self.net.get_priority(node, dst, max_neigh=MAX_NEIGH)
                                          ,node, mcs, priority_dict={}, loss_dict={})
                neighbours = self.net.egress_neigh(node, dst, MAX_NEIGH, mcs)
                for neigh in neighbours:
                    loss = self.net.get_err(node, neigh, mcs)
                    # to avoid exorbitant high results when using high securities
                    if loss < MAX_LOSS:
                        coop_group[node].add_neighbour(neigh, self.net.get_priority(neigh, dst), loss)
            self.dst_coop_groups[dst] = coop_group
        if not initialising:
            self.update_link_paths()

    # removes node from cooperation groups with 2 nodes and one of them is the dst
    # should only be used if node failures and not only link failures are assumed
    def remove_2node_dst_neigh(COOP_GROUPS):
        for node in COOP_GROUPS.values():
            if len(node.priority_dict) == 2 and max(node.priority_dict.values()) == float('inf'):
                node.rm_neighbour(min(node.priority_dict, key=node.priority_dict.get))
        return COOP_GROUPS

    def get_links_on_path(self, source, dst):
        path = (source, dst)
        link_list = [link for link in self.links if path in self.links[link].get_participating_paths()]
        return link_list

    def get_nodes_on_path(self, source, dst):
        """
        Returns a list of nodes that are on the path between source and destination
        :param source:
        :param dst:
        :return:list of strings
        """
        node_list = []
        for link in self.get_links_on_path(source, dst):
            link_source, link_dst = link
            if link_source not in node_list and link_source is not source:
                node_list.append(link_source)
        return node_list

    def get_node_names(self):
        return list(self.dst_coop_groups.keys())


def dump_network(network):
    np.save("network.npy", network)


def load_network(path="network.py"):
    return np.load(path).item()


def main():
    network = Network()
    network.set_next_loss_window()
    network.set_next_loss_window()
    network.update_coop_groups()


if __name__ == '__main__':
    main()
