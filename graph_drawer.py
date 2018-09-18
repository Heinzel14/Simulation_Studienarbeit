from prioritycruncher import *
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv
from global_variables import MAX_NEIGH

net = Network(crunched_root='net_dump_priority.npy')
for dst in ['fl-geek']: #net.nodes:
    G = nx.DiGraph()
    # adding nodes to graph
    if MAX_NEIGH:
        for node in net.nodes:
            if node == dst:
                net.set_max_node_priority(node, dst, MAX_NEIGH, float('inf'))
                G.add_node(node, style='filled', fillcolor='red')
            else:
                net.set_max_node_priority(node, dst, MAX_NEIGH, 0)
                G.add_node(node)
    net.compute_priority(dst, max_rounds=10, max_neigh=MAX_NEIGH, handle_correlation=False)


    for node in net.nodes:
        for link in net.egress_links(node, dst, MAX_NEIGH, net.ideal_coop_mcs(node, dst, MAX_NEIGH, False)):
            mcs = net.ideal_coop_mcs(node, dst, MAX_NEIGH, False)
            mcs_bitmask = link['mcs_bitmasks'][str(mcs)]['bitmap']
            G.add_edge(link['src'], link['dst'], penwidth=
                str(10*net.get_link_err(link, net.ideal_coop_mcs(node, dst, MAX_NEIGH, False)))) # str(0.05*mcs_dr_20Mhz_GI[mcs] *(np.count_nonzero(mcs_bitmask)) / len(mcs_bitmask))) #
            if link['dst'] == dst:
                print(link['src'], mcs_dr_20Mhz_GI[mcs] *(np.count_nonzero(mcs_bitmask)) / len(mcs_bitmask))
    mapping = {nodename: (nodename + ' ' + str(round(net.get_priority(nodename, dst, MAX_NEIGH), 1))) for nodename in net.nodes}
    nx.relabel_nodes(G, mapping, copy=False)
    # set defaults
    G.graph['graph']={'rankdir':'TD'}
    G.graph['node']={'shape':'circle'}
    G.graph['edges']={'arrowsize':'4.0'}


    A = to_agraph(G)
    print(A)
    A.layout('dot')
    data_name = dst+'new_pf_test.png'
    A.draw(data_name)

