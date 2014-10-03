import sys

from camip import MatrixNetlist, CAMIP, VPRSchedule
import numpy as np
import pandas as pd
from cyplace_experiments.data import open_netlists_h5f


def place(net_file_namebase):
    netlists_h5f = open_netlists_h5f()
    netlist_group = getattr(netlists_h5f.root.netlists, net_file_namebase)
    connections = pd.DataFrame(netlist_group.connections[:])
    block_type_labels = netlist_group.block_type_counts.cols.label[:]
    netlist = MatrixNetlist(connections,
                            block_type_labels[netlist_group.block_types[:]])
    placer = CAMIP(netlist)
    placer.shuffle_placement()
    print placer.evaluate_placement()
    schedule = VPRSchedule(placer.s2p, 1, placer.netlist, placer)
    print 'starting temperature: %.2f' % schedule.anneal_schedule.temperature
    costs = schedule.run(placer)
    print costs[-1]
    return placer, costs


if __name__ == '__main__':
    #import matplotlib.pyplot as plt

    if len(sys.argv) == 3:
        np.random.seed(int(sys.argv[2]))
    placer, costs = place(sys.argv[1])
    #fig = plt.figure()
    #ax = fig.add_subplot(111, title=sys.argv[1])
    #ax.plot(costs)
    #plt.show(block=True)
