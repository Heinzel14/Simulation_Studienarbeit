from network import Network
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv
from global_variables import MAX_NEIGH
from global_variables import MAX_LOSS
from mesh_simulator_without_kodo import calc_tot_send_time
import numpy as np
import matplotlib.colors

def draw_forwarding_network(source, dst):
    network = Network()
    #cmap = matplotlib.colors.get_named_colors_mapping()
    cmap = plt.get_cmap('gnuplot')
        #LinearSegmentedColormap.from_list("", ["red","green", "blue"])
    G = nx.DiGraph()
    # adding nodes to graph
    for node in network.get_nodes_on_path(source, dst):
        G.add_node(node)
    calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=True)

    for link in network.get_links_on_path(source, dst):
        link_source, link_dst = link
        penwidth = 1+10*network.get_link_flow((link_source, link_dst))
        color = matplotlib.colors.rgb2hex(cmap(network.get_link_flow((link_source, link_dst))))
        G.add_edge(link_source, link_dst, penwidth=penwidth, color=color)

    mapping = {nodename: nodename for nodename in network.get_node_names()}
    nx.relabel_nodes(G, mapping, copy=False)
    # set defaults
    G.graph['graph'] = {'rankdir': 'TD'}
    G.graph['node'] = {'shape': 'circle'}
    G.graph['edges'] = {'arrowsize': '4.0'}
    A=to_agraph(G)
    A.layout('dot')
    data_name = dst + '_fair_pf.png'
    A.draw(data_name)


def main():
    draw_forwarding_network('mirko', 'stud2')


if __name__ == '__main__':
    main()


