from collections import deque
from global_variables import *
import coefficients_getter as cg
from network import Network
import numpy as np



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


def calc(buffer, network, dst, greedy_mode=False, fair_forwarding=False, single_path=False):

    # this simulates a node failure | not needed anymore as node can be set to failed in network now
    # if coop_groups[node['name']].name in fail_nodes:
    #     return 0buffer = deque([])

    node = buffer.popleft()
    coop_groups = network.get_dst_coop_groups(dst)
    priorities = coop_groups[node['name']].get_priorities()
    expected_losses = coop_groups[node['name']].get_losses()
    datarate = coop_groups[node['name']].get_datarate()

    # pf and c calculation are chosen accordingly to the strategy
    if single_path:
        c, pf_Dict = cg.get_single_path_parameters(priorities, expected_losses)

    elif greedy_mode:
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
        # print('got dst feedback')
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
        m = min(min(n * (1 - link_loss), node['m']) * pf_Dict[neigh], 1)
        network.add_link_data((node['name'], neigh), m)
        # print(node['name'], neigh, network.get_link_flow((node['name'], neigh)))
        if len(coop_groups[neigh].get_priorities()) > 0 and pf_Dict[neigh] != 0:
            # first min: if loss is lower and neigh receives more there are only m innovative packets
            # second min: there can not be more innovative packets than the generation size
            buffer.append({'name': neigh, 'm': m})

    return sending_time


def calc_tot_send_time(network, source, dst, greedy_mode=False, fair_forwarding=False, single_path=False):

    network.reset_link_forwardings()
    buffer = deque([])
    tot_send_time = 0
    coop_groups = network.get_dst_coop_groups(dst)

    # should only be used if node and not only edge failures are assumed
    # if greedy_mode == True:
    #     notwork.remove_True2node_dst_neigh(coop_groups)

    buffer.append({'name': source, 'm': 1})
    while len(buffer) > 0:
        if coop_groups[dst].total_data_received >= 0.99999:
            break
        tot_send_time += calc(buffer, network, dst, greedy_mode, fair_forwarding, single_path)
    # this is the resending
    try:
        if coop_groups[dst].total_data_received < 0.99999:
            tot_send_time = tot_send_time * 1/coop_groups[dst].total_data_received
    except:
        print('single path mistalke')
        print(source, dst)


    if coop_groups[dst].total_data_received > 1.0001:
        print('Destination got', coop_groups[dst].total_data_received)
        raise NameError("Source feedback did not work properly. Source got more than one generation size")

    network.reset_vertex_counter()
    return tot_send_time


def compare_filter_rules():
    #set priorities to first window
    network = Network()
    network.set_next_loss_window()
    network.update_coop_groups()
    counter = int(MIN_BITMAP_SIZE/WINDOW_SIZE)-1
    average_send_time_dict = {'normal':[], 'ff':[], 'ff_fe':[], 'sp':[]}
    for i in range(counter):
        print('calculating for window ', i, 'out of ', counter)
        network.set_next_loss_window()
        send_times = {'normal': [], 'ff': [], 'ff_fe': [], 'sp': []}
        for dst in network.get_node_names():
            for source in network.get_node_names():

                if source == dst:
                    continue

                if not network.way_to_dst(source, dst) or not network.single_path_way_to_dst(source,dst):
                    print('no connection for ', source, dst)
                    continue
                send_times['normal'].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                               fair_forwarding= False))
                send_times['ff'].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                           fair_forwarding=True))
                send_times['ff_fe'].append(calc_tot_send_time(network, source, dst, greedy_mode=True,
                                                              fair_forwarding=True))
                if network.single_path_way_to_dst(source, dst):
                    send_times['sp'].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                              fair_forwarding=False, single_path=True))
        average_send_time_dict['normal'].append(np.mean(send_times['normal']))
        average_send_time_dict['ff'].append(np.mean(send_times['ff']))
        average_send_time_dict['ff_fe'].append(np.mean(send_times['ff_fe']))
        average_send_time_dict['sp'].append(np.mean(send_times['sp']))


    return average_send_time_dict


def compare_filter_rules_with_edge_failures(max_failures):
    network = Network()
    send_times = {'normal': {}, 'ff': {}, 'ff_fe': {}}
    for dst in network.get_node_names():
        print('calculating for dst', dst)

        for source in network.get_node_names():
            if source == dst:
                continue
            for i in range(max_failures):
                print('calculating for',i,'failed edges')
                for links in combinations(network.get_links_on_path(source, dst), i):
                    for link in links:
                        network.set_link_failure(link)
                    if not network.way_to_dst(source, dst):
                        network.reset_failures()
                        continue
                    if str(i) not in send_times['normal']:
                        send_times['normal'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                           fair_forwarding=False)]
                    else:
                        send_times['normal'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                               fair_forwarding=False))
                    if str(i) not in send_times['ff']:
                        send_times['ff'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                       fair_forwarding=True)]
                    else:
                        send_times['ff'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                           fair_forwarding=True))
                    if str(i) not in send_times['ff_fe']:
                        send_times['ff_fe'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=True,
                                                                          fair_forwarding=True)]
                    else:
                        send_times['ff_fe'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=True,
                                                                              fair_forwarding=True))
                    network.reset_failures()
    return send_times


def compare_filter_rules_with_node_failures(max_failures):
    network = Network()
    send_times = {'normal': {}, 'ff': {}, 'ff_fe': {}}
    for dst in network.get_node_names():
        print('calculating for dst', dst)

        for source in network.get_node_names():
            if source == dst:
                continue
            for i in range(max_failures):
                print('calculating for',i,'failed nodes')
                for nodes in combinations(network.get_nodes_on_path(source, dst), i):
                    for node in nodes:
                        network.set_node_failure(node)
                    if not network.way_to_dst(source, dst):
                        network.reset_failures()
                        continue
                    if str(i) not in send_times['normal']:
                        send_times['normal'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                           fair_forwarding=False)]
                    else:
                        send_times['normal'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                               fair_forwarding=False))
                    if str(i) not in send_times['ff']:
                        send_times['ff'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                       fair_forwarding=True)]
                    else:
                        send_times['ff'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=False,
                                                                           fair_forwarding=True))
                    if str(i) not in send_times['ff_fe']:
                        send_times['ff_fe'][str(i)] = [calc_tot_send_time(network, source, dst, greedy_mode=True,
                                                                          fair_forwarding=True)]
                    else:
                        send_times['ff_fe'][str(i)].append(calc_tot_send_time(network, source, dst, greedy_mode=True,
                                                                              fair_forwarding=True))
                    network.reset_failures()
    return send_times





def main():
    np.save("send_time_filter_rules_over_time_no_failures.npy", compare_filter_rules())
    # np.save("send_time_filter_rules_node_failures.npy", compare_filter_rules_with_node_failures(4))
    # np.save("send_fime_filter_rules_failures.npy",compare_filter_rules_with_edge_failures(4))






if __name__ == '__main__':
    main()


