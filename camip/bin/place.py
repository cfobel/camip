import sys

import numpy as np
from camip import MatrixNetlist, CAMIP
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

    placer.evaluate_placement()

    for i in xrange(10):
        placer.run_iteration(np.random.randint(10000), 1)

    return placer


if __name__ == '__main__':
    placer = place(sys.argv[1])
