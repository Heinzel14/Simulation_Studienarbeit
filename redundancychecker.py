from prioritycruncher import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import network as vc
#from mesh_simulator import calc_tot_send_time
from collections import deque
from global_variables import *


# variable to check for errors. should always be bigger than 1 after a alpha calculation
currentDstData = 0
plotData = []
additional_sending_data_dict = {'testmode':{}, 'no testmode':{}}
COOP_GROUPS = {}

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


def calculate_c(losses, failureNodes, FilterAdaption, pfDict = []):
    """
    calculates the sending coefficient according to the redundancy we need to add to compensate the given failure nodes
    :param losses: dict of nodes and the losses on the ede to them
    :param pfDict: Dictionary of filter coefficients
    :return: sending coefficient
    """
    # adaption for pfs that got 0
    for node in pfDict:
        if pfDict[node] == 0 and node not in failureNodes:
            failureNodes.append(node)

    if FilterAdaption == False:
        c = 0
        lossCopy = copy.deepcopy(losses)
        for failureNode in failureNodes:
            if failureNode in lossCopy:
                del lossCopy[failureNode]
        for node in lossCopy:
            c += (1 - lossCopy[node]) * pfDict[node]
        return c
    else:
        totalFailure = 1
        lossCopy = copy.deepcopy(losses)
        for failureNode in failureNodes:
            if failureNode in lossCopy:
                del lossCopy[failureNode]
        for loss in lossCopy.values():
            totalFailure *= loss
        return 1 - totalFailure

def calc_testmode_pfs(losses, priorities):
    pf_dict = {}
    loss_stack = deque([])
    priorities_copy = copy.deepcopy(priorities)
    predecessors = []
    # find the destination in the coop group
    blNeighbour = get_bl_neighbour(losses, priorities)
    c = calculate_c(losses, blNeighbour, True)

    # create loss stack beginning with highest priority neigh
    for i in range(len(priorities)):
        max_neigh = max(priorities_copy, key=priorities_copy.get)
        loss_stack.append((max_neigh, losses[max_neigh]))
        del priorities_copy[max_neigh]

    # if dst in coop group adapt situation to dst (can not fail so the next to pfs are the errer of the dst and the c is different)
    # the dst is also not part of the predecessors as the c is adapted
    if max(priorities.values()) == float('inf'):
        dst = max(priorities, key=priorities.get)
        c = c - (1-losses[dst])
        pf_dict[dst] = 1
        loss_stack.popleft()
        for i in range(2):
            if len(loss_stack) == 0:
                return pf_dict
                return pf_dict
            name, loss = loss_stack.popleft()
            predecessors.append(((1-loss)*losses[dst]))
            pf_dict[name] = losses[dst]
            pred_loss = loss
            pred_name = name

    else:
        for i in range(2):
            if len(loss_stack) == 0:
                return pf_dict
            name, loss = loss_stack.popleft()
            pf = 1
            # this part is not neede as in the mesh simulator we forward min(m, (1-e)n)
            # if (1-losses[name]) > c:
            #     # result of the formula for a node that gets more than c
            #     pf = c/(1-losses[name])
            predecessors.append((1-loss)*pf)
            pf_dict[name] = pf
            pred_loss = loss
            pred_name = name


    while len(loss_stack) > 0:
        total_forwarded = sum(del_max_pred(predecessors))
        name, loss = loss_stack.popleft()

        # node steals a little bit of the forwarded data of the predecossor
        if total_forwarded >= c:
            # data that should be stolen is dependant on the loss of the predocessor
            steal_data = pred_loss*predecessors[-1]

            # if the data that should be stolen is bigger than tha data the node gets
            if steal_data > (1-losses[name]):
                pf = 1
                pf_dict[name] = pf
                steal_data = (1-losses[name])
            else:
                pf = steal_data/(1-loss)
                pf_dict[name] = pf

            pf_dict[pred_name] = (predecessors[-1]-steal_data)/(1-pred_loss)
            pred_loss = loss
            pred_name = name
            predecessors.append((1-loss)*pf)


            # if nothing is left to forward nodes are sending nothing (Attention: c would havo to be adapted)
            # while len(loss_stack) > 0:
            #
            #     name, loss = loss_stack.popleft()
            #
            #     pf_dict[name] = 0
            # return pf_dict
        else:
            # node should forward what would still be lost in the wc scenario if it can else it forwards what it gets
            pf = (c-total_forwarded)/(1-loss)
            if pf >1:
                pf = 1
                pf_dict[name] = 1
            else:
                pf_dict[name] = (c - total_forwarded) / (1 - loss)
            predecessors.append((1-loss)*pf)
            pred_loss = loss
            pred_name = name

    return pf_dict


def give_c(losses, dst=None):

    """
    :param losses: list of losses
    :param dst: loss value of the destination
    :return: average success ratio with the best link failing (except the source)
    """

    loss_copy = copy.deepcopy(losses)
    if dst:
        loss_copy.remove(dst)
        loss_copy.remove(min(loss_copy))
        loss_copy.append(dst)
    else:
        loss_copy.remove(min(loss_copy))
    e_tot = 1
    for i in loss_copy:
        e_tot *= i
    return 1 - e_tot


def rest_to_res_c(loss_dict, pfs_dict, c, dst=None):
    losses = []
    pfs = []
    pfs_dict_copy = copy.deepcopy(pfs_dict)
    if dst:
        del pfs_dict_copy[dst]

    for key in pfs_dict_copy:
        losses.append(loss_dict[key])
        pfs.append(pfs_dict[key])

    forwarded = [(1 - a) * b for a, b in zip(losses, pfs)]
    forwarded.remove(max(forwarded))
    return (c - sum(forwarded))


def calc_fair_pfs(loss_dict, priority_dict):
    c = give_c(loss_dict.values())
    pfs = {node: 0 for node in loss_dict}
    open_nodes = len(pfs)
    rest = 0
    dst = None
    if max(priority_dict.values()) == float('inf'):
        dst = max(priority_dict, key=priority_dict.get)
        pfs[dst] = 1
        open_nodes -= 1
        if len(pfs) == 1:
            return pfs
        c = give_c(loss_dict.values(), loss_dict[dst]) - (1 - loss_dict[dst])

    c_part = c / open_nodes

    # if only one node set 1 and return
    if len(pfs) == 1:
        for key in pfs:
            pfs[key] = 1
        return pfs


    # initialisation (first round)
    for node in pfs:

    #this is the source
        if pfs[node] == 1:
            continue

        pf = c_part / (1 - loss_dict[node])
        if pf > 1:
            pfs[node] = 1
            rest += (c_part - (1 - loss_dict[node]))
            open_nodes -= 1
        else:
            pfs[node] = pf
    rest += rest_to_res_c(loss_dict, pfs, c, dst)

    # fill up the nodes until lower bound is reached
    while rest > 0.000001 * c:

        c_part = rest / open_nodes
        rest = 0
        for node in pfs:
            if pfs[node] == 1:
                continue
            pf = c_part / (1 - loss_dict[node])
            # adding new part to old one
            if pf + pfs[node] > 1:
                pfs[node] = 1
                # 1-pfs[i] is what it can forward until it reaches 1
                rest += (c_part - (1 - pfs[node]) * (1 - loss_dict[node]))
                open_nodes -= 1
            else:
                pfs[node] += pf
        rest += rest_to_res_c(loss_dict, pfs, c, dst)
    return pfs


def pf_tester(pf_dict, losses, c, priorities):
    forwarding_sum = 0
    max_forwarded = 0
    # if c == 0:
    #     print('zero?',pf_dict, losses)
    bl_neighbour, wc_neighbour = get_bl_wc_neighbour(priorities, losses, testmode = True)
    for node in pf_dict:
        if pf_dict[node] < 0:
            print ('negative pf')
            print (pf_dict)
            print (losses)
            return False
        node_forwarding = (1-losses[node])*pf_dict[node]
        if node not in wc_neighbour or len(losses)==1:
            forwarding_sum += node_forwarding
    if c/forwarding_sum > 1.00001:
        print('forwarding sum to small', forwarding_sum, c)
        print('loss dicionary',losses)
        return False
    else:
        if forwarding_sum/c > 1.1:
            print('we send to much')
            print(losses)
            print(priorities)
            print(pf_dict)
        return True


# returns copy of predecessors list without the one forwarding the most
def del_max_pred(predecessors):
    predecessors = copy.deepcopy(predecessors)
    max_forward_stuff = 0
    max_forward_index = None
    for index in range(len(predecessors)):
        if predecessors[index] > max_forward_stuff:
            max_forward_stuff = predecessors[index]
            max_forward_index = index
    del predecessors[max_forward_index]
    return predecessors


def calculate_pf(losses, priorities, failureNodes=[], testmode=False):
    """
    This function is calculating the filter coefficients
    :param losses: dictionary {nodename: loss}
    :param priorities: dictionary {nodename: priority}
    :param failureNodes: you need to give failure Nodes for an Adaption of the filtercoefficients
    :return: list of filter coefficients
    """
    pf = []
    namesSorted = []
    lossesSortedPrio = []
    prio = copy.deepcopy(priorities)
    i = len(priorities) - 1
    j = len(priorities)

    # make sorted list for the losses of the vertices with decreasing priority
    for x in range(i):
        maxNeighName = max(prio, key=prio.get)
        namesSorted.append(maxNeighName)
        lossesSortedPrio.append(losses[maxNeighName])
        del prio[maxNeighName]
    # add worst node in the name list (his loss is not needed for the filter coefficients) to create the dictionary in the end
    maxNeighName = max(prio, key=prio.get)
    namesSorted.append(maxNeighName)
    parentsLosses = []
    # if testmode == True:
    #
    #     for x in range(j):
    #         # add losses of nodes with higher priority before current node
    #         if x > 0:
    #             parentsLosses.append(lossesSortedPrio[x - 1])
    #         # the first two pfs are 1
    #         if x == 0 or x == 1 and max(priorities.values()) != float('inf'):
    #             pf.append(1)
    #         # after that we are taking the worst combinations of the error rates of the nodes before
    #         if x == 2 or x== 1 and max(priorities.values()) == float('inf'):
    #             maxcombination = 0
    #             # if dst is in coop group we start one iteration earlier with the losses
    #             if max(priorities.values()) == float('inf'):
    #                 for combination in combinations(parentsLosses, x):
    #                     if np.prod(combination) > maxcombination:
    #                         maxcombination = np.prod(combination)
    #             else:
    #                 for combination in combinations(parentsLosses, x-1):
    #                     if np.prod(combination) > maxcombination:
    #                         maxcombination = np.prod(combination)
    #             pf.append(maxcombination)
    #     pfDict = dict(zip(namesSorted, pf))
    #     return pf, pfDict
    if testmode == True:
        pfDict = calc_testmode_pfs(losses, priorities)
        return pfDict.values(), pfDict

    else:
        # calculate filter coefficients
        for x in range(j):
            # first one is always 1
            if x == 0:
                pf.append(1)
            # if node before can break down we are taking its coefficient to compensate loss (only works if c is also adapted!)
            elif namesSorted[x - 1] in failureNodes:
                pf.append(pf[x - 1])
            # here we apply the normal formula for calculating the filter coefficients
            else:
                pf.append(pf[x - 1] * lossesSortedPrio[x - 1])
        pfDict = dict(zip(namesSorted, pf))
        return pf, pfDict


def get_greedy_stategy_pfs(priorities, losses):
    blNeighbour = get_bl_neighbour(losses, priorities)
    # pfDictA = calc_testmode_pfs(losses, priorities) #calculate_pf(losses, priorities, wcNeighbour, testmode=True)

    pfDictA = calc_testmode_pfs(losses,priorities) #calc_fair_pfs(losses, priorities)
    cA = calculate_c(losses, blNeighbour, True, pfDictA)

    if pf_tester(pfDictA, losses, cA, priorities) == False:
        print('pf dict', pfDictA)
        print('priority dict',priorities)
        raise NameError('pfs are not forwarding enough')

    return (pfDictA.values(), pfDictA, cA)
    # pfListNA, pfDictNA = calculate_pf(losses, priorities)
    # cNA = calculate_c(losses, wcNeighbour, False, pfDictNA)
    # dataNA = (float(1) / cNA) + ((float(1) / cNA) * sum(pfListNA))
    # dataA = (float(1) / cA) + ((float(1) / cA) * sum(pfListA))
    # if dataNA > dataA:
    #     pfList, pfDict = pfListA, pfDictA
    #     c = cA
    #     # print('Adaption')
    #     # plotData.append(dataNA - dataA)
    # elif dataNA < dataA:
    #     pfList, pfDict = pfListNA, pfDictNA
    #     c = cNA
    #     print('NO Adaption')
    #     #plotData.append(dataNA - dataA)
    # # this is only the case if there is only the destination the cooperation group (no failure to prevent --> both strategies giving the same value)
    # elif dataA == dataNA:
    #     pfList, pfDict = pfListA, pfDictA
    #     c = cA
    # return(pfList, pfDict, c)


def get_bl_neighbour(losses, priorities):
    lowestLoss = 1
    if max(priorities.values()) == float('inf'):
        dst = max(priorities, key=priorities.get)
    else:
        dst = None
    # can not compensate if there is only 1 neighbour
    if len(losses) == 1:
        blNeighbour = []
    else:
        for node in losses:
            # looking for node that has the lowest (best) loss and is not the dst (cant break down)
            if losses[node] < lowestLoss and node != dst:
                lowestLoss = losses[node]
                blNeighbour = [node]
    return blNeighbour



def get_bl_wc_neighbour(priorities, losses, testmode = False):
    """
    this function searches for the neighbour in the priorities dict which is forwarding the most and
    the neighbour with the lowest loss in the loss dict
    :param priorities:
    :param losses:
    :return:
    """
    pfList, pfDict = calculate_pf(losses, priorities, [], testmode=testmode)
    worstValue = 0
    lowestLoss = 1
    # can not compensate if there is only 1 neighbour
    if len(priorities) == 1:
        wcNeighbour = []
        blNeighbour = []
    else:
        for node in pfDict:
            # looking for node that is forwarding the most (not the dst because it is not forwarding anything)
            if pfDict[node] * (1 - losses[node]) > worstValue and priorities[node] != float('inf'):
                worstValue = pfDict[node] * (1 - losses[node])
                wcNeighbour = [node]
            # looking for node that has the lowest (best) loss and is not the dst (cant break down)
            if losses[node] < lowestLoss and priorities[node] != float('inf'):
                lowestLoss = losses[node]
                blNeighbour = [node]
    return blNeighbour, wcNeighbour

def calculate_data(source, pf, loss, data, failureNodes=[], sender='start Point', FilterAdaption=False, testnodes=[],
                   testmode=False):
    """
    This function is calculating the data per paket in the network starting from the source to the destintaion
    which is given because of the priorities in the global variable cooperationGroFilterAdaption=True
    :param source: String that contains the name of the starting node
    :param pf: filter coefficient this node should apply
    :param loss: loss on the data which was send to this node
    :param data: data that was send to this node
    :param sender: String that contains the name of the node which was sending data to the source
    :param failureNodes:
    :return: total redundancy we send per paket
    """
    global plotData
    global currentDstData
    global additional_sending_data_dict
    global COOP_GROUPS
    myFailureNodes = copy.deepcopy(failureNodes)
    priorities = copy.deepcopy(COOP_GROUPS[source].get_priorities())
    losses = copy.deepcopy(COOP_GROUPS[source].get_losses())
    # when there are no neighbours, the vertice won't send anything (we reached the destination)
    if len(priorities) == 0:
        currentDstData += (1 - loss) * data
        return 0
    # if node is a failure node it is not forwarding
    if len(testnodes) > 0:  # failureNodes) > 0:
        if source in testnodes:
            return 0
    # if source in failureNodes:
    # 	return 0
    # myFailureNodes = []
    # sometimes a node gets data with the value 0 because of error coefficients of 0
    if data == 0:
        return 0
    if testmode == True:
        blNeighbour, wcNeighbour = get_bl_wc_neighbour(priorities, losses, testmode=True)
        # pfList, pfDict = calculate_fair_pf(losses, blNeighbour, priorities)
        # c= calculate_c(losses, blNeighbour, True)
        pfList, pfDict, c = get_greedy_stategy_pfs(priorities, losses)

    if testmode == False:
        if FilterAdaption == True:
            pfList, pfDict = calculate_pf(losses, priorities, myFailureNodes)
        else:
            pfList, pfDict = calculate_pf(losses, priorities, [])
        c = calculate_c(losses, myFailureNodes, FilterAdaption, pfDict)
    if c == 0:
        return 0
    # data sent by the node is normalized to the datarate to evaluate time
    myData = (pf * ((1 - loss) * data) / c)
    sendingTime = float(myData) / COOP_GROUPS[source].get_datarate()
    # here is the recursive part of the function. The total data send in the network beginning from each neighbour is added
    # each node is normalizing its sent data to the datarate bit gives only the sent data to its neighbours
    # so they can calculate their forwarded and apply their own datarate
    
    if testmode == True:
        if source not in additional_sending_data_dict['testmode']:
            additional_sending_data_dict['testmode'][source] = {'sending time': myData, 'pf': [pf]}
        else:
            additional_sending_data_dict['testmode'][source]['sending time'] += myData# myData#*sum([pfDict[node]*(1-losses[node]) for node in pfDict])
            additional_sending_data_dict['testmode'][source]['pf'].append(pf)
    if testmode == False:
        if source not in additional_sending_data_dict['no testmode']:
            additional_sending_data_dict['no testmode'][source] = {'sending time': myData, 'pf': [pf]}
        else:
            additional_sending_data_dict['no testmode'][source]['sending time'] += myData# myData#*sum([pfDict[node]*(1-losses[node]) for node in pfDict])
            additional_sending_data_dict['no testmode'][source]['pf'].append(pf)

    for pfVertice in pfList:
        maxNeighName = max(priorities, key=priorities.get)
        loss = losses[maxNeighName]
        del priorities[maxNeighName]
        sendingTime += calculate_data(maxNeighName, pfVertice, loss, myData, failureNodes, source, FilterAdaption,
                                     testnodes=testnodes, testmode=testmode)
    return sendingTime


# # at the moment there is only the 'greedy strategy' for 1 failure node
# def calculate_no_feedback_alphas(maxNodeFailures=1, window=False):
#     global MIN_BITMAP_SIZE
#     global WINDOW_SIZE
#     global COOP_GROUPS
#     netalphas = {}
#     dstalphas = {}
#
#     if window == True:
#         windowCounter = int(MIN_BITMAP_SIZE / WINDOW_SIZE)
#         for Node in COOP_GROUPS:
#             COOP_GROUPS[Node].set_next_window()
#     elif window == False:
#         windowCounter = 1
#     dst_coop_groups = vc.load_dst_coop_groups()
#     for dst in dst_coop_groups:
#         print('calculating for dst ', dst)
#         COOP_GROUPS = dst_coop_groups[dst]
#
#         # calculating for every window (error rates are changed accordingly)
#         for window in range(windowCounter):
#             priorities = []
#             sortednodes = []
#             # print('calculating for window', window)
#
#             # creating sorted list of node priorities
#             for node in COOP_GROUPS:
#                 priorities.append(COOP_GROUPS[node].priority)
#             priorities.sort()
#
#             # creating sorted list of nodes (by priorities)
#             for priority in priorities:
#                 for node in COOP_GROUPS:
#                     if COOP_GROUPS[node].priority == priority:
#                         sortednodes.append(COOP_GROUPS[node].name)
#
#             # we need at least two nodes between source and dst
#             for i in range(len(sortednodes) - 2):
#                 alphas = []
#                 source = sortednodes[i]
#                 print (source,dst)
#                 m0 = calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=False, failnodes=[], source_feedback= True)
#                 print ('m0 is ', m0)
#                 m_res = calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=True, failnodes=[], source_feedback= True)
#                 print ('m resilent is ', m_res)
#                 alpha = (m_res/m0)-1
#                 if alpha < 0:
#                     print (source, dst, alpha)
#                     raise NameError('negative alpha ')
#                 alphas.append(alpha) # calculate_data(source, 1, 0, 1, testmode=True) / m0) - 1)
#                 if len(alphas) > 0:
#                     if (len(sortednodes)-i, maxNodeFailures, window) not in netalphas:
#                         netalphas[(len(sortednodes)-i, maxNodeFailures, window)] = alphas
#                     else:
#                         netalphas[(len(sortednodes)-i, maxNodeFailures, window)] += alphas
#                 dstalphas[(dst,source)] = alpha # ((calculate_data(source, 1, 0, 1, testmode=True) / m0) - 1)
#
#             # update window for next iteration
#             for Node in COOP_GROUPS:
#                 COOP_GROUPS[Node].set_next_window()
#     print('finished alpha calculation')
#     return netalphas, dstalphas
#
#
# def calculate_feedback_alphas(maxNodeFailures, window=False, FilterAdaption=False):
#     global currentDstData
#     global MIN_BITMAP_SIZE
#     global WINDOW_SIZE
#     global COOP_GROUPS
#     netalphas = {}
#
#     if window == True:
#         windowCounter = int(MIN_BITMAP_SIZE/WINDOW_SIZE)
#         for Node in COOP_GROUPS:
#             COOP_GROUPS[Node].set_next_window()
#     elif window == False:
#         windowCounter = 1
#
#     dst_coop_groups = vc.load_dst_coop_groups()
#     for dst in dst_coop_groups:
#         print('calculating for dst ', dst)
#         COOP_GROUPS = dst_coop_groups[dst]
#         # calculating for every window (error rates are changed accordingly)
#         for window in range(windowCounter):
#             priorities = []
#             sortednodes = []
#             # print('calculating for window', window)
#
#             # creating sorted list of node priorities
#             for node in COOP_GROUPS:
#                 priorities.append(COOP_GROUPS[node].priority)
#             priorities.sort()
#
#             # creating sorted list of nodes (by priorities)
#             for priority in priorities:
#                 for node in COOP_GROUPS:
#                     if COOP_GROUPS[node].priority == priority:
#                         sortednodes.append(COOP_GROUPS[node].name)
#
#             # we need at least two nodes between source and dst
#             for i in range(len(sortednodes) - 2):
#                 source = sortednodes[i]
#                 # print('source is', source)
#                 currentDstData = 0
#                 m0 = calculate_data(source, 1, 0, 1)
#                 possibleFailureNodes = sortednodes[i + 1:len(sortednodes) - 1]
#                 participationNodes = sortednodes[i:len(sortednodes) - 1]
#                 for n in range(1, maxNodeFailures+1):
#                     alphas = []
#                     for failureNodes in combinations(possibleFailureNodes, n):
#                         if len(failureNodes) == 0:
#                             break
#                         length = -1
#                         failureNodesCopy = copy.deepcopy(failureNodes)
#
#                         # searching for disconnected nodes and add them to failure nodes
#                         while length != len(failureNodesCopy):
#                             length = len(failureNodesCopy)
#                             for participationNode in participationNodes:
#                                 neighbours = COOP_GROUPS[participationNode].get_priorities()
#                                 neighbours = neighbours.keys()
#                                 if participationNode not in failureNodesCopy and \
#                                         len(set(failureNodesCopy) & set(neighbours)) == len(neighbours):
#                                     failureNodesCopy.append(participationNode)
#                                     if participationNode == source:
#                                         break
#
#                         # node is skipped if connection loss occurs
#                         if source in failureNodesCopy:
#                             # alphas.append('looser')
#                             continue
#                         currentDstData = 0
#
#                         # for testing this is a no feedback alpha --> should be changed afterwards
#                         if len(failureNodesCopy) == 1:
#                             alpha = (calc_tot_send_time(source, dst, fr_mode=False, greedy_mode=True, failnodes=failureNodesCopy) / m0) - 1
#
#                             alphas.append(alpha)
#
#                         # alphas.append((calculate_data(source, 1, 0, 1, failureNodesCopy, FilterAdaption=FilterAdaption, testmode=False) / m0)- 1)
#                         # if currentDstData < 0.9999:
#                         #     print('error destination got not 1. data, source, dst, window and loss dict')
#                         #     print(currentDstData)
#                         #     print(source, dst, window)
#                         #     print(COOP_GROUPS[source].get_losses())
#
#                     if len(alphas) > 0:
#                         if (len(participationNodes), n, window) not in netalphas:
#                             netalphas[(len(participationNodes), n, window)] = alphas
#                         else:
#                             netalphas[(len(participationNodes), n, window)] += alphas
#
#             # update window for next iteration
#             for Node in COOP_GROUPS:
#                 COOP_GROUPS[Node].set_next_window()
#
#     return netalphas


def bardrawer(netalphas, nmax, maxnetsize):
    counter = 0
    for netsize in range(2, maxnetsize + 1):
        for n in range(1, nmax + 1):
            if (netsize, n) in netalphas:
                plt.figure(counter)
                drawstuff = []
                for element in netalphas[(netsize, n)]:
                    if len(drawstuff) < element:
                        while len(drawstuff) < element - 1:
                            drawstuff.append([])
                        drawstuff.append([element])
                    else:
                        drawstuff[element - 1].append(element)
                y = [0]
                for element in drawstuff:
                    y.append((len(element) / len(netalphas[netsize, n])))
                while len(y) < 5:
                    y.append(0)
                N = len(y)
                x = [0]
                for i in range(1, N):
                    x.append(i)
                width = 0.4
                plt.bar(x, y, width, color="red")
                plt.xlabel('actual failurenodes', size='x-large')
                plt.ylabel('percentage', size='x-large')
                title = 'Actual failures for ' + str(n) + ' removed nodes '
                plt.title(title,size='x-large')
                counter += 1
                dataname = str(netsize + 2) + '-' + str(n)
                plt.savefig(dataname, bbox_inches='tight')
                plt.close()


def drawer(netalphas, nmax, maxnetsize=None):
    patches = [mpatches.Patch(color='blue', label='1 failure node'),
               mpatches.Patch(color='orange', label='2 failure node'),
               mpatches.Patch(color='green', label='3 failure node')]
    # for netsize in range(2, maxnetsize + 1):
    #     plt.figure(netsize - 1)
    #xmax = 0
    for n in range(1, nmax + 1):
        data = []
        for key in netalphas:
            netnow, nnow = key
            if n == nnow:
                data += netalphas[key]
        datasorted = sorted(data)
        print(data)
        print(netalphas)
        #xmax = max(xmax, max(data))
        p = 1. * np.arange(len(datasorted)) / (len(datasorted) - 1)
        plt.plot(datasorted, p, label='n=' + str(n), lw=3)
    plt.xlabel('alpha',size='x-l arge')
    plt.ylabel('cdf',size='x-large')
    title = 'CDF of alpha'  # n=1 to ' + str(nmax) + ' failure nodes and a netsize of ' + str(netsize + 2) + ' nodes'
    plt.title(title, size='x-large')
    axes = plt.gca()
    # if xmax > 10:
    #  	xmax=10
    #axes.set_xlim([0, xmax])
    axes.set_ylim([0, 1])
    #plt.legend(loc=4)
    # dataname = str(netsize + 2) + 'nodes'
    # plt.savefig(dataname, bbox_inches='tight')
    # plt.close()
    plt.show()


def main():
    global COOP_GROUPS
    # net = Network(data_root='test_data/2017-7-31-11-35-46')
    # net = Network(crunched_root='net_dump_priority.npy')
    #net = Network(crunched_root='net_dump_priority.npy')
    # netalphas = calculate_feedback_alphas(1)

    netalphas, dst_alphas = calculate_no_feedback_alphas()
    # print(currentDstData)
    # calculate_data(source, 1, 0, 1, testmode=True)
    #
    #
    # for node in net.nodes:
    #     try:
    #         print(node, (additional_sending_data_dict['testmode'][node]/ additional_sending_data_dict['no testmode'][node]))
    #         print(additional_sending_data_dict['testmode'][node])
    #         print(additional_sending_data_dict['no testmode'][node])
    #     except:
    #         continue
    np.save("new_pfs_alphas.npy", dst_alphas)

if __name__ == '__main__':
    main()