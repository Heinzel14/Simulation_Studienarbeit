from collections import deque
from global_variables import *
import coefficients_getter as cg
from network import Network

buffer = deque([])


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


def calc(node, network, dst, greedy_mode=False, fair_forwarding=False):

    global buffer

    # this simulates a node failure | not needed anymore as node can be set to failed in network now
    # if coop_groups[node['name']].name in fail_nodes:
    #     return 0

    coop_groups = network.get_dst_coop_groups(dst)
    priorities = coop_groups[node['name']].get_priorities()
    expected_losses = coop_groups[node['name']].get_losses()
    datarate = coop_groups[node['name']].get_datarate()

    # pf and c calculation are chosen accordingly to the strategy
    if greedy_mode:
        pf_List, pf_Dict, c = cg.get_greedy_stategy_pfs(priorities, expected_losses)

    elif fair_forwarding:
        pf_Dict = cg.calc_fair_pfs(expected_losses, priorities, False)
        c = cg.calc_c(expected_losses, [])
    else:
        pf_List, pf_Dict = cg.calc_pf(expected_losses, priorities)
        c = cg.calc_c(expected_losses, [])

    m = node['m']
    test_value = sum([pf_Dict[name]*(1-expected_losses[name]) for name in pf_Dict])/c
    if test_value > 1.0000000000001 and greedy_mode == False:
        raise NameError('there should be no redundancy forwarded without greedy mode or '
                        'not enough data forwarded')

    n = m/c

    # this simulates an instant dst feedback if full rank. The losses here are the actual ones not the
    # expected because it is a feedback
    if dst and dst in pf_Dict and coop_groups[dst].total_data_received + n * \
            (1 - network.get_ideal_link_loss(node['name'], dst, dst)) >= 1:
        print('got dst feedback')
        n = (1/(1-network.get_ideal_link_loss(node['name'], dst, dst)))\
            * (1 - coop_groups[dst].total_data_received)
        m = n*c
    coop_groups[node['name']].total_data_sent += m
    sending_time = n/datarate

    for neigh in pf_Dict:
        link_loss = network.get_ideal_link_loss(node['name'], neigh, dst)
        coop_groups[neigh].total_data_received += n*(1-link_loss)
        # only nodes with neighbours are added to the buffer (dst or other nodes with
        # empty coop group won't send)
        if len(coop_groups[neigh].get_priorities()) > 0 and pf_Dict[neigh] != 0:
            # first min: if loss is lower and neigh receives more there are only m innovative packets
            # second min: there can not be more innovative packets than the generation size
            m = min(min(n*(1-link_loss), node['m'])*pf_Dict[neigh], 1)
            buffer.append({'name': neigh, 'm': m})

    return sending_time


def calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=False):
    global buffer
    tot_send_time = 0
    coop_groups = network.get_dst_coop_groups(dst)

    # should only be used if node and not only edge failures are assumed
    # if greedy_mode == True:
    #     notwork.remove_True2node_dst_neigh(coop_groups)

    buffer.append({'name': source, 'm': 1})
    while len(buffer) > 0:
        if coop_groups[dst].total_data_received >= 0.99999:
            break
        tot_send_time += calc(buffer.popleft(), network, dst, greedy_mode, fair_forwarding)
    # this is the resending
    if coop_groups[dst].total_data_received < 0.99999:
        tot_send_time = tot_send_time * 1/coop_groups[dst].total_data_received

    if coop_groups[dst].total_data_received > 1.0001:
        print('Destination got', coop_groups[dst].total_data_received)
        raise NameError("Source feedback did not work properly. Source got more than one generation size")

    network.reset_vertex_counter()
    return tot_send_time


def main():
    global tot_m
    dst = 'frawi'
    source = 'kitchen'
    network = Network()
    # network.set_node_failure('fl_geek')
    send_time = calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=False)
    print(send_time, 'sending time')
    paths = network.get_links_on_path(source, dst)
    print(len(paths))
    for i in combinations(paths):
        for link in i:
            network.set_link_failure(link)
        if not network.way_to_dst(source, dst):
            print(network.way_to_dst(source, dst))
        network.reset_failures()
    # send_time = calc_tot_send_time(network, source, dst, greedy_mode=True, fair_forwarding=True)
    # print(send_time, 'sending time')
    # for i in range(30):
    #     network.set_next_loss_window()
    # send_time = calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=False)
    # print(send_time, 'sending time')
    # network.update_coop_groups()
    # send_time = calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=False)
    # print(send_time, 'sending time')


if __name__ == '__main__':
    main()


