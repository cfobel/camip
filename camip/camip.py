'''
# Concurrent Associated-Moves Iterative Placement (CAMIP) #

This file contains a reference implementation of the CAMIP model described in
Christian Fobel's PhD thesis.

__NB__ This implementation is _not_ optimized for run-time performance.  It is
intended to provide a concise, matrix-based implementation of the CAMIP model,
which should easily be ported to libraries supporting sparse-matrix operations.

## License ##

    Copyright (C) 2014 Christian Fobel <christian@fobel.net>

    This file is part of CAMIP.

    CAMIP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    CAMIP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CAMIP.  If not, see <http://www.gnu.org/licenses/>.
'''
from __future__ import division
import itertools
import pandas as pd
import numpy as np
from scipy.stats import itemfreq
import scipy.sparse as sparse

from cyplace_experiments.data import open_netlists_h5f
from .CAMIP import (evaluate_moves, VPRMovePattern, VPRAutoSlotKeyTo2dPosition,
                    Extent2D, slot_moves, extract_positions)

try:
    profile
except:
    profile = lambda (f): f


class MatrixNetlist(object):
    def __init__(self, connections, block_types):
        CLOCK_PIN = 5
        self.connections = connections
        self.global_nets = (connections[connections['pin_key'] ==
                                        CLOCK_PIN]['net_key'].unique())
        # Filter out connections that correspond to a global net.
        self.local_connections = connections[~connections['net_key']
                                             .isin(self.global_nets)]

        self.block_types = block_types
        self.block_count = self.local_connections['block_key'].unique().size
        self.block_type_counts = dict([(k, int(v)) for k, v in
                                       itemfreq(self
                                                .block_types[:self
                                                             .block_count])])
        #for i, (k, t) in self.local_connections[['block_key',
                                                 #'block_type']].iterrows():
            #if self.block_types[k] is None:
                #self.block_types[k] = t
                #self.block_type_counts[t] = (self.block_type_counts
                                             #.setdefault(t, 0) + 1)

        self.P = sparse.coo_matrix((self.local_connections['pin_key'],
                                    (self.local_connections['block_key'],
                                     self.local_connections['net_key'])),
                                   dtype=np.uint8)
        self.C = sparse.coo_matrix((np.ones(len(self.local_connections)),
                                    (self.local_connections['block_key'],
                                     self.local_connections['net_key'])),
                                   dtype=np.uint8)
        self.r = self.C.astype(np.uint32).sum(axis=0)
        self.net_ones = np.matrix([np.ones(self.C.shape[0])], dtype=np.uint32)
        assert((self.net_ones * self.C == self.r).all())


def random_pattern_params(max_magnitude, extent, non_zero=True):
    max_magnitude = min(extent - 1, max_magnitude)
    magnitude = np.random.randint(0, max_magnitude + 1)
    while non_zero and magnitude == 0:
        magnitude = np.random.randint(0, max_magnitude + 1)

    shift = 0

    if magnitude > 0:
        max_shift = 2 * magnitude - 1
        if extent <= 2 * magnitude:
            max_shift = magnitude - 1
        shift = np.random.randint(max_shift + 1)

    return magnitude, shift


def random_2d_pattern_params(max_magnitude, vpr_s2p):
    max_magnitude.row = min(vpr_s2p.extent.row - 1, max_magnitude.row)
    max_magnitude.column = min(vpr_s2p.extent.column - 1, max_magnitude.column)

    magnitude = Extent2D()
    magnitude.row = np.random.randint(0, max_magnitude.row + 1)
    magnitude.column = np.random.randint(0, max_magnitude.column + 1)

    while (magnitude.row == 0) and (magnitude.column == 0):
        magnitude.row = np.random.randint(0, max_magnitude.row + 1)
        magnitude.column = np.random.randint(0, max_magnitude.column + 1)

    shift = Extent2D()

    if magnitude.row > 0:
        max_shift = 2 * magnitude.row - 1
        if vpr_s2p.extent.row <= 2 * magnitude.row:
            max_shift = magnitude.row - 1
        shift.row = np.random.randint(max_shift + 1)

    if magnitude.column > 0:
        max_shift = 2 * magnitude.column - 1
        if vpr_s2p.extent.column <= 2 * magnitude.column:
            max_shift = magnitude.column - 1
        shift.column = np.random.randint(max_shift + 1)
    return magnitude, shift


def random_vpr_pattern(vpr_s2p, max_logic_move=None, max_io_move=None):
    io_extent = vpr_s2p.slot_count.io
    if max_io_move is None:
        max_io_move = io_extent - 1
    io_move, io_shift = random_pattern_params(max_io_move, io_extent)
    if max_logic_move is None:
        max_logic_move = Extent2D(vpr_s2p.extent.row - 1,
                                  vpr_s2p.extent.column - 1)
    logic_move, logic_shift = random_2d_pattern_params(max_logic_move, vpr_s2p)
    return VPRMovePattern(io_move, io_shift, logic_move, logic_shift, vpr_s2p)


class CAMIP(object):
    def __init__(self, netlist):
        self.netlist = netlist
        self.io_count = (netlist.block_type_counts['.input'] +
                         netlist.block_type_counts['.output'])
        self.logic_count = netlist.block_type_counts['.clb']
        self.s2p = VPRAutoSlotKeyTo2dPosition(self.io_count, self.logic_count,
                                              2)
        self.slot_block_keys = np.empty(len(self.s2p), dtype=np.uint32)
        self.slot_block_keys[:] = netlist.block_count

        # Fill IO slots.
        self.slot_block_keys[:self.io_count] = [i for i, t in
                                                enumerate(netlist.block_types)
                                                if t in ('.input',
                                                         '.output')][:self
                                                                     .io_count]
        # Fill logic slots.
        logic_start = self.s2p.slot_count.io
        logic_end = logic_start + self.logic_count
        self.slot_block_keys[logic_start:logic_end] = [i for i, t in
                                                       enumerate(netlist
                                                                 .block_types)
                                                       if t in ('.clb')]

        # Create reverse-mapping, from each block-key to the permutation slot
        # the block occupies.
        self.block_slot_keys = np.empty(netlist.block_count, dtype=np.uint32)
        self.block_slot_moves = np.empty(netlist.block_count, dtype=np.int32)
        self.block_slot_keys_prime = np.empty_like(self.block_slot_keys)
        self._sync_slot_block_keys_to_block_slot_keys()

        self.p_x = np.empty(netlist.block_count, dtype=np.int32)
        self.p_y = np.empty(netlist.block_count, dtype=np.int32)
        self.p_x_prime = np.empty(netlist.block_count, dtype=np.int32)
        self.p_y_prime = np.empty(netlist.block_count, dtype=np.int32)

        self.Y = netlist.C._with_data(np.empty_like(netlist.C.data,
                                                    dtype=np.float32),
                                      copy=False)
        self.X = netlist.C._with_data(np.empty_like(netlist.C.data,
                                                    dtype=np.float32),
                                      copy=False)
        self.omega = netlist.C._with_data(np.empty_like(netlist.C.data,
                                                        dtype=np.float32),
                                          copy=False)
        self.omega_prime = netlist.C._with_data(np.empty_like(netlist.C.data,
                                                              dtype=np
                                                              .float32),
                                                copy=False)
        self.r_inv = np.reciprocal(netlist.r.astype(np.float32))

    def shuffle_placement(self):
        '''
        Shuffle placement permutation.

        The shuffle is aware of IO and logic slots in the placement, and will
        keep IO and logic within the corresponding areas of the permutation.
        '''
        np.random.shuffle(self.slot_block_keys[:p.s2p.slot_count.io])
        np.random.shuffle(self.slot_block_keys[p.s2p.slot_count.io:])
        self._sync_slot_block_keys_to_block_slot_keys()

    def _sync_block_slot_keys_to_slot_block_keys(self):
        '''
        Update `slot_block_keys` based on `block_slot_keys`.

        Useful when updating the permutation slot contents directly _(e.g.,
        shuffling the contents of the permutation slots)_.
        '''
        self.slot_block_keys[:] = self.netlist.block_count
        self.slot_block_keys[self.block_slot_keys] = xrange(self.netlist
                                                            .block_count)

    def _sync_slot_block_keys_to_block_slot_keys(self):
        '''
        Update `block_slot_keys` based on `slot_block_keys`.
        '''
        occupied = np.where(self.slot_block_keys < self.netlist.block_count)
        self.block_slot_keys[self.slot_block_keys[occupied]] = occupied[0]
        del occupied

    def evaluate_placement(self):
        '''
        Compute the cost of:

         - Each net _(`self.e_c`)_.
         - The complete placement _(`self.theta`)_.
        '''
        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments.
        extract_positions(self.p_x, self.p_y, self.block_slot_keys, self.s2p)

        self.X.data[:] = self.p_x[self.X.row]
        self.Y.data[:] = self.p_y[self.Y.row]

        # Star+ vectors
        self.e_x = self.X.sum(axis=0)
        self.e_x2 = self.X.multiply(self.X).sum(axis=0)
        self.e_y = self.Y.sum(axis=0)
        self.e_y2 = self.Y.multiply(self.Y).sum(axis=0)
        self.e_c = 1.59 * (np.sqrt(np.square(self.e_x) - self.e_x2.A *
                                   self.r_inv.A + 1) +
                           np.sqrt(np.square(self.e_y) - self.e_y2.A *
                                   self.r_inv.A + 1))

        # `theta`: $\theta =$ total placement cost
        self.theta = self.e_c.sum()

        # `omega`: $\Omega \in \mathbb{M}_{mn}$, $\omega_{ij} = e_cj$.
        self.omega.data[:] = map(self.e_c.A.ravel().__getitem__,
                                 self.netlist.C.col)

        # $\vec{n_c}$ contains the total cost of all edges connected to node
        # $i$.
        self.n_c = self.omega.sum(axis=1)
        return self.theta

    @profile
    def propose_moves(self, seed, max_io_move=None, max_logic_move=None):
        np.random.seed(seed)
        self.move_pattern = random_vpr_pattern(self.s2p,
                                               max_io_move=max_io_move,
                                               max_logic_move=max_logic_move)
        slot_moves(self.block_slot_moves, self.block_slot_keys,
                   self.move_pattern)
        self.block_slot_keys_prime[:] = (self.block_slot_keys +
                                         self.block_slot_moves)

        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments.
        extract_positions(self.p_x_prime, self.p_y_prime,
                          self.block_slot_keys_prime, self.s2p)

    @profile
    def evaluate_moves(self):
        # __NB__ Use Cython function _(improves run-time performance by almost
        # three orders of magnitude)_.
        evaluate_moves(self.omega_prime.data, self.omega_prime.row,
                       self.omega_prime.col,
                       self.p_x, self.p_x_prime, self.e_x.A.ravel(),
                       self.e_x2.A.ravel(),
                       self.p_y, self.p_y_prime, self.e_y.A.ravel(),
                       self.e_y2.A.ravel(),
                       self.r_inv.A.ravel(), 1.59)

        self.n_c_prime = self.omega_prime.sum(axis=1)
        self.delta_n = self.n_c_prime - self.n_c

    def assess_groups(self, temperature):
        total_slot_count = len(self.s2p)
        self.block_group_keys = np.fromiter((min(k1, k2) if k1 != k2 else
                                             total_slot_count
                                             for k1, k2 in
                                             itertools
                                             .izip(self.block_slot_keys,
                                                   self
                                                   .block_slot_keys_prime)),
                                            dtype=np.int32)
        moved_mask = (self.block_slot_keys != self.block_slot_keys_prime)
        unmoved_count = moved_mask.size - moved_mask.sum()

        group_block_keys = np.argsort(self.block_group_keys)[:-unmoved_count]
        if len(group_block_keys) == 0:
            self.delta_s = np.empty(0)
            return 0, np.empty(0)
        packed_group_segments = np.empty_like(group_block_keys)
        packed_group_segments[0] = 1
        packed_group_segments[1:] = (self.block_group_keys
                                     [group_block_keys][1:] !=
                                     self.block_group_keys
                                     [group_block_keys][:-1])
        packed_block_group_keys = np.cumsum(packed_group_segments) - 1

        self.S = sparse.coo_matrix((self.delta_n[group_block_keys].A.ravel(),
                                    (group_block_keys,
                                     packed_block_group_keys)))

        self.delta_s = self.S.sum(axis=0).A.ravel()

        assess_urands = np.random.rand(len(self.delta_s))
        a = ((self.delta_s <= 0) | (assess_urands < np.exp(-self.delta_s /
                                                           temperature)))
        rejected_block_keys = self.S.row[~a[packed_block_group_keys]]
        return (moved_mask.size - unmoved_count), rejected_block_keys

    def apply_groups(self, rejected_move_block_keys):
        if len(rejected_move_block_keys) == 0:
            return
        self.block_slot_keys_prime[rejected_move_block_keys] = (
            self.block_slot_keys[rejected_move_block_keys])
        self.block_slot_keys, self.block_slot_keys_prime = (
            self.block_slot_keys_prime, self.block_slot_keys)
        self._sync_block_slot_keys_to_slot_block_keys()

    @profile
    def run_iteration(self, seed, temperature, max_io_move=None,
                      max_logic_move=None):
        self.propose_moves(seed, max_io_move=max_io_move,
                           max_logic_move=max_logic_move)
        self.evaluate_moves()
        moved_count, rejected_move_block_keys = self.assess_groups(temperature)
        self.apply_groups(rejected_move_block_keys)
        self.evaluate_placement()
        return moved_count, rejected_move_block_keys


def main():
    connections = pd.DataFrame(np.array([[0, 0, 0],
                                         [1, 1, 0],
                                         [2, 0, 0], [2, 2, 4],
                                         [3, 0, 0], [3, 1, 1], [3, 7, 2], [3, 3, 4],
                                         [4, 0, 0], [4, 1, 1], [4, 4, 4],
                                         [5, 2, 0], [5, 3, 1], [5, 5, 4], [5, 10, 5],
                                         [6, 3, 0], [6, 6, 4],
                                         [7, 4, 0], [7, 7, 4], [7, 10, 5],
                                         [8, 5, 0], [8, 0, 1], [8, 8, 4],
                                         [9, 6, 0], [9, 7, 1], [9, 9, 4], [9, 10, 5],
                                         [10, 8, 0],
                                         [11, 6, 0],
                                         [12, 9, 0], [13, 10, 0]]),
                               columns=['block_key', 'net_key', 'pin_key'])
    return MatrixNetlist(connections, 2 * ['.input'] + 8 * ['.clb'] + 3 *
                         ['.output'] + ['.input'])


if __name__ == '__main__':
    netlist = main()
    p = CAMIP(netlist)

    netlists_h5f = open_netlists_h5f()
    netlist_group = netlists_h5f.root.netlists.ex5p
    connections = pd.DataFrame(netlist_group.connections[:])
    block_type_labels = netlist_group.block_type_counts.cols.label[:]
    ex5p = MatrixNetlist(connections,
                         block_type_labels[netlist_group.block_types[:]])
