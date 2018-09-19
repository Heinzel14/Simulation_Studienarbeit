import copy
def calc_pf(losses, priorities, failureNodes=[]):
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
    # add worst node in the name list (his loss is not needed for the filter coefficients) to
    # create the dictionary in the end
    maxNeighName = max(prio, key=prio.get)
    namesSorted.append(maxNeighName)
    # calculate filter coefficients
    for x in range(j):
        # first one is always 1
        if x == 0:
            pf.append(1)
        # if node before can break down we are taking its coefficient
        # to compensate loss (only works if c is also adapted!)
        elif namesSorted[x - 1] in failureNodes:
            pf.append(pf[x - 1])
        # here we apply the normal formula for calculating the filter coefficients
        else:
            pf.append(pf[x - 1] * lossesSortedPrio[x - 1])
    pfDict = dict(zip(namesSorted, pf))
    return pf, pfDict

def calc_c(losses, failureNodes):
    """
    calculates the sending coefficient removing the links to the nodes given in failureNodes
    :param losses: dict of nodes and the losses on the ede to them
    :param pfDict: Dictionary of filter coefficients
    :return: sending coefficient
    """
    # adaption for pfs that got 0
    # for node in pfDict:
    #     if pfDict[node] == 0 and node not in failureNodes:
    #         failureNodes.append(node)
    totalFailure = 1
    lossCopy = copy.deepcopy(losses)
    for failureNode in failureNodes:
        if failureNode in lossCopy:
            del lossCopy[failureNode]
    for loss in lossCopy.values():
        totalFailure *= loss
    return 1 - totalFailure

def calc_res_c(losses, dst=None):

    """
    removes the smallest loss except the dst and calculates the average sending ratio needed
    :param losses: list of losses
    :param dst: loss value of the destination
    :return: average success ratio with the best link failing (except the source)
    """
    loss_list = list(losses.values())
    if dst and len(losses) > 1:
        dst_loss = losses[dst]
        loss_list.remove(dst_loss)
        loss_list.remove(min(loss_list))
        loss_list.append(dst_loss)
    elif len(losses) > 1:
        loss_list.remove(min(loss_list))
    e_tot = 1
    for loss in loss_list:
        e_tot *= loss
    return 1 - e_tot


def get_wc_neighbour(priorities, losses, pf_dict):

    """
    this function searches for the neighbour in the priorities dict which is forwarding the most and
    the neighbour with the lowest loss in the loss dict
    :param priorities:
    :param losses:
    :return:
    """
    worst_value = 0
    # can not compensate if there is only 1 neighbour
    if len(priorities) == 1:
        wc_neighbour = []
    else:
        for node in pf_dict:
            # looking for node that is forwarding the most (also dst because a link failure to dst has to be
            # compensated)
            if pf_dict[node] * (1 - losses[node]) > worst_value:  # and priorities[node] != float('inf'):
                worst_value = pf_dict[node] * (1 - losses[node])
                wc_neighbour = [node]
    return wc_neighbour


def rest_to_c(loss_dict, pfs_dict, c, dst=False, resilient=False):
    """
    calculates how much more data has to be forwarded to compensate the failure of the node forwarding the most
    (lower bound)
    :param loss_dict:
    :param pfs_dict:
    :param c:
    :param dst:
    :param resilient:
    :return:
    """
    losses = []
    pfs = []
    pfs_dict_copy = copy.deepcopy(pfs_dict)
    if dst:
        print('deleting dst in rest to resilent c')
        del pfs_dict_copy[dst]

    for key in pfs_dict_copy:
        losses.append(loss_dict[key])
        pfs.append(pfs_dict[key])

    zip(losses, pfs)
    forwarded = [(1 - a) * b for a, b in zip(losses, pfs)]
    if resilient:
        forwarded.remove(max(forwarded))
    return c - sum(forwarded)


def calc_fair_pfs(loss_dict, priority_dict, greedy_mode=True):
    """
    calculates the forwarding coefficients evenly distributed, so that the amount of additional sent data is
    minimized
    :param loss_dict:
    :param priority_dict:
    :param greedy_mode:
    :return:
    """
    c = calc_c(loss_dict, [])
    if greedy_mode:
        c = calc_res_c(loss_dict)
    pfs = {node: 0 for node in loss_dict}
    open_nodes = len(pfs)
    rest = 0

    # set destination pf to 1
    if max(priority_dict.values()) == float('inf'):
        # Attention! here we assume dst node can not fail but path to it can --> link failure
        dst = max(priority_dict, key=priority_dict.get)
        pfs[dst] = 1
        open_nodes -= 1
        if len(pfs) == 1:
            return pfs

    # if only one node set 1 and return
    if len(pfs) == 1:
        for key in pfs:
            pfs[key] = 1
        return pfs

    rest += rest_to_c(loss_dict, pfs, c, resilient=greedy_mode)
    # fill up the nodes until lower bound is reached
    while rest > 0.000001 * c:

        c_part = rest/open_nodes

        rest = 0
        for node in pfs:
            if pfs[node] == 1:
                continue
            pf = c_part / (1 - loss_dict[node])
            # adding new part to old one
            if pf + pfs[node] > 1:
                pfs[node] = 1
                open_nodes -= 1
            else:
                pfs[node] += pf

        rest += rest_to_c(loss_dict, pfs, c, resilient=greedy_mode)

    pf_tester(pfs, loss_dict, c, priority_dict, greedy_mode)
    return pfs


def pf_tester(pf_dict, losses, c, priorities, greedy_mode):
    forwarding_sum = 0
    wc_neighbour = get_wc_neighbour(priorities, losses, pf_dict)

    for node in pf_dict:
        if pf_dict[node] < 0:
            print(pf_dict)
            raise NameError('negative pfs')

        node_forwarding = (1-losses[node])*pf_dict[node]
        if greedy_mode and node not in wc_neighbour or len(losses) == 1:
            forwarding_sum += node_forwarding
        if not greedy_mode:
            forwarding_sum += node_forwarding

    if c/forwarding_sum > 1.0001:
        print('failure report:')
        print('c and forwarding sum', c, forwarding_sum)
        print('pfs', pf_dict)
        print('losses', losses)
        print('priorities', priorities)
        print('greedy mode', greedy_mode)
        raise NameError('Total forwarded to small')

    elif forwarding_sum/c > 1.1:
        print('failure report:')
        print('c and forwarding sum', c, forwarding_sum)
        print('pfs', pf_dict)
        print('losses', losses)
        print('priorities', priorities)
        print('greedy mode', greedy_mode)
        raise NameError('forwarding to much')
    return


def get_greedy_stategy_pfs(priorities, losses):
    """
    calculates the forwarding coefficients for the resilient strategy
    :param priorities: dictionary of node priorities
    :param losses: dictionary of node losses
    :return: list and dictionary of the forwarding coefficients and the success ratio assuming the best link
    failing
    """
    pf_dict = calc_fair_pfs(losses, priorities, True)
    c = calc_res_c(losses, None)  # get_dst_loss(priorities, losses))
    return pf_dict.values(), pf_dict, c


def get_dst_name(priority_dict):
    if max(priority_dict.values()) == float('inf'):
        return max(priority_dict, key=priority_dict.get)
    else:
        return None


def get_dst_loss(priority_dict, loss_dict):
    if max(priority_dict.values()) == float('inf'):
        return loss_dict[max(priority_dict, key=priority_dict.get)]
    else:
        return None


def get_bl_neighbour(losses, priorities):
    """
    :param losses:
    :param priorities:
    :return: list with one element which is the name of the neighbour forwarding the most, except the source
    """
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