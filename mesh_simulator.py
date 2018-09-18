import copy
import sys
import kodo
import random
import os
from collections import deque
import network as vc
from global_variables import *
import coefficients_getter as cg
from redundancychecker import get_greedy_stategy_pfs

dst_counter = 0

transmission_counter = 0
forwarding_counter = 0
buffer = deque([])
tot_send_time = 0
tot_m = 0

encoder_factory = kodo.RLNCEncoderFactory(kodo.field.binary16, SYMBOLS, SYMBOL_SIZE)
decoder_factory = kodo.RLNCDecoderFactory(kodo.field.binary16, SYMBOLS, SYMBOL_SIZE)
recoder_factory = kodo.RLNCPureRecoderFactory(kodo.field.binary16, SYMBOLS, SYMBOL_SIZE)
recoder_factory.set_recoder_symbols(100)


def get_err_list(transmission_number, err):
    err_ellements = int(transmission_number*err)
    err_list = ['loss'] * err_ellements + ['success'] * (transmission_number-err_ellements)
    random.shuffle(err_list)
    return err_list


def calc(node, coop_groups, fr_mode, greedy_mode, fail_nodes=[], dst = None):
    """
    :param node: dictionary that contains m,
    :return:
    """

    global transmission_counter
    global tot_send_time
    global buffer
    global forwarding_counter
    global tot_m
    global dst_counter
    # nodes that are already full rank should not forward anymore
    # if coop_groups[node['name']].full_rank == True and fr_mode == True:
    #     return coop_groups
    # simulating node failure
    if coop_groups[node['name']].name in fail_nodes:
        return coop_groups


    priorities = copy.deepcopy(coop_groups[node['name']].get_priorities())
    losses = copy.deepcopy(coop_groups[node['name']].get_losses())
    datarate = coop_groups[node['name']].get_datarate()


    # pf and c calculation are chosen accordingly to the strategy
    if greedy_mode == True:
        pf_List, pf_Dict, c = cg.get_greedy_stategy_pfs(priorities, losses)
        # pf_List, pf_Dict, c = get_greedy_stategy_pfs(priorities, losses)


        #blNeighbour, wcNeighbour = get_bl_wc_neighbour(priorities, losses)
        #pf_List, pf_Dict = calculate_fair_pf(losses, blNeighbour, priorities)
    else:
        pf_List, pf_Dict = cg.calc_pf(losses, priorities)
        c = cg.calc_c(losses, [])

    m = node['m']

    test_value = sum([pf_Dict[name]*(1-losses[name]) for name in pf_Dict])/c
    if test_value > 1.0000000000001 and greedy_mode == False:
        raise NameError('there should be no redundancy forwarded without greedy mode or '
                        'not enough data forwarded')


    # if node is sending more than generation size it only sends enough to reach it an than stops
    # this is wrong! we need to send more because m is not received by one node
    # if coop_groups[node['name']].total_data_sent > SYMBOL_SIZE*SYMBOLS and fr_mode == True:
    #     print ('AAAAAIIUUUUTOOOOOOOOO')
    #     m = m-(coop_groups[node['name']].total_data_sent - SYMBOLS * SYMBOL_SIZE)
    #     coop_groups[node['name']].full_rank = True

    n = m/c

    # this simulates an instant dst feedback if full rank
    if dst and dst in pf_Dict and coop_groups[dst].total_data_received + n * (1 - losses[dst]) >= SYMBOLS * SYMBOL_SIZE:
        print ('got dst feedback')
        n = (1/(1-losses[dst]))*(SYMBOL_SIZE*SYMBOLS - coop_groups[dst].total_data_received)
        m = n*c
    coop_groups[node['name']].total_data_sent += m
    tot_m += m
    tot_send_time += n/datarate
    coop_groups[node['name']].sending_rank += int(m / SYMBOL_SIZE)


    Gs = SYMBOL_SIZE*SYMBOLS
    sending_counter = int((n + SYMBOL_SIZE) / SYMBOL_SIZE)

    err_lists = {}
    for neigh in pf_Dict:
        coop_groups[neigh].total_data_received += n*(1-losses[neigh])
        err_lists[neigh] = get_err_list(sending_counter, losses[neigh])
        # only nodes with neighbours are added to the buffer (dst or other nodes with
        # empty coop group won't send)
        if len(coop_groups[neigh].get_priorities()) > 0 and pf_Dict[neigh]!=0:
            m = min(min(n*(1-losses[neigh]), node['m'])*pf_Dict[neigh], Gs)
            buffer.append({'name':neigh, 'm':m})

    for i in range(sending_counter):
        packet = coop_groups[node['name']].coder.write_payload()
        for neigh in err_lists:
            if err_lists[neigh][i] == 'success':
                coop_groups[neigh].coder.read_payload(packet)
                coop_groups[neigh].packet_counter+=1

    transmission_counter += 1
    forwarding_counter += 1/c * node['m']
    return coop_groups


    # # after sending check if node is full rank --> nodes that got full rank should forward one more time
    # if COOP_GROUPS[node['name']].sending_rank > 100:
    #     COOP_GROUPS[node['name']].full_rank = True
    #     #print (node['name'], 'got full rank')

# apply changes is an resend time function
def calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=False, failnodes=[], source_feedback = True):
    global buffer
    global tot_send_time
    global transmission_counter
    transmission_counter = 0

    tot_send_time = 0
    dst_coop_groups = vc.load_dst_coop_groups()
    coop_groups = dst_coop_groups[dst]


    for vertice in coop_groups:
        if vertice == source:
            coop_groups[vertice].set_source(encoder_factory)
        elif vertice == dst:
            coop_groups[vertice].coder = decoder_factory.build()
            data_out = bytearray(coop_groups[vertice].coder.block_size())
            coop_groups[vertice].coder.set_mutable_symbols(data_out)
        else:
            coop_groups[vertice].coder = recoder_factory.build()
    # should only be used if node and not only edge failures are assumed
    # if greedy_mode == True:
    #     vc.remove_2node_dst_neigh(coop_groups)

    buffer.append({'name': source, 'm': (SYMBOLS * SYMBOL_SIZE)})
    while len(buffer) > 0:
        if coop_groups[dst].total_data_sent >= SYMBOLS*SYMBOL_SIZE and source_feedback==True:
            break
        calc(buffer.popleft(), coop_groups, fr_mode, greedy_mode, failnodes, dst)
    # because of limited Gs coder rank will not always get full --> only checking for very low ranks to
    # identify problems
    try:
        if coop_groups[dst].coder.rank() < 80:
            raise NameError('not full rank')
    except NameError:
        print('did not get full rank!!!')
        print(source, dst, coop_groups[dst].coder.rank(), coop_groups[dst].packet_counter, failnodes)
        # print(coop_groups[dst].sending_rank)
        # print('fr_mode, greedy_mode', fr_mode, greedy_mode)
        # print(transmission_counter, tot_send_time)
    return tot_send_time  # , coop_groups[dst].coder.rank())


def calc_tot_resend_time(source, dst, fr_mode=False, greedy_mode=False, failnodes=[]):
    global buffer
    global tot_send_time
    global transmission_counter
    transmission_counter = 0

    tot_send_time = 0
    dst_coop_groups = vc.load_dst_coop_groups(fair=False)
    coop_groups = dst_coop_groups[dst]

    for vertice in coop_groups:
        if vertice == source:
            coop_groups[vertice].set_source(encoder_factory)
        elif vertice == dst:
            coop_groups[vertice].coder = decoder_factory.build()
            data_out = bytearray(coop_groups[vertice].coder.block_size())
            coop_groups[vertice].coder.set_mutable_symbols(data_out)
        else:
            coop_groups[vertice].coder = recoder_factory.build()
    # should only be used if node and not only edge failures are assumed
    # if greedy_mode == True:
    #     vc.remove_True2node_dst_neigh(coop_groups)

    buffer.append({'name': source, 'm': (SYMBOLS * SYMBOL_SIZE)})
    while len(buffer) > 0:
        if coop_groups[dst].total_data_received >= SYMBOLS * SYMBOL_SIZE:
            break
        coop_groups = calc(buffer.popleft(), coop_groups, fr_mode, greedy_mode, failnodes, dst)
        print coop_groups[dst].total_data_received
    if coop_groups[dst].total_data_received < (SYMBOLS * SYMBOL_SIZE):
        GS = SYMBOL_SIZE*SYMBOLS
        buffer.append({'name': source, 'm': GS*(GS/coop_groups[dst].total_data_received-1)})
        print ('resending ', GS*(GS/coop_groups[dst].total_data_received-1), coop_groups[dst].coder.rank())
    while len(buffer) > 0:
        coop_groups = calc(buffer.popleft(), coop_groups, fr_mode, greedy_mode, failnodes, dst)
        print coop_groups[dst].total_data_received
    print('dst got ',coop_groups[dst].total_data_received, coop_groups[dst].coder.rank())
    print 'attention'
    return tot_send_time  # , coop_groups[dst].coder.rank())


def main():
    global tot_m
    dst = 'frawi'
    source = 'kitchen'
    # source= 'fl-geek'coop_groups[dst].total_data_sent
    send_time = calc_tot_resend_time(source, dst, fr_mode=False, greedy_mode=False, failnodes=['fl-geek'])
    # print(transmission_counter, 'total transmissions')
    # print(forwarding_counter)
    print(send_time,'sending time')


if __name__ == '__main__':
    main()








