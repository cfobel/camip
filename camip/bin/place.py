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

    total_moves = np.zeros(100, dtype=int)
    for i in xrange(100):
        non_zero_moves, rejected = placer.run_iteration(np.random
                                                        .randint(10001), 1)
        total_moves[i] = non_zero_moves

    return placer, total_moves


if __name__ == '__main__':
    placer, total_moves = place(sys.argv[1])
