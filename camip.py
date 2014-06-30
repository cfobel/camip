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
import math
import pandas as pd
import numpy as np
from scipy.stats import itemfreq
import scipy.sparse as sparse

from cyplace_experiments.data import open_netlists_h5f


class SlotKeyTo2dPosition(object):
    def __init__(self, row_extent, offset=None):
        if offset is None:
            self.offset = {'x': 0, 'y': 0}
        else:
            self.offset = offset
        self.row_extent = row_extent

    def __getitem__(self, k):
        p = {'x': int(k // self.row_extent), 'y': int(k % self.row_extent)}
        return {'x': int(p['x'] + self.offset['x']),
                'y': int(p['y'] + self.offset['y'])}


class VPRIOSlotKeyTo2dPosition(object):
    def __init__(self, extent, io_capacity):
        self.extent = extent
        self.io_capacity = io_capacity

        io_count = {'row': self.extent['row'] * self.io_capacity,
                    'column': self.extent['column'] * self.io_capacity}
        self.segment_start = {'bottom': 0, 'right': io_count['row']}
        self.segment_start['top'] = (self.segment_start['right'] +
                                     io_count['column'])
        self.segment_start['left'] = (self.segment_start['top'] +
                                      io_count['row'])
        self._size = 2 * (io_count['column'] + io_count['row'])

    def __len__(self):
        return self._size

    def __getitem__(self, k):
        if k < self.segment_start['right']:
            # Slot `k` maps to position along the 'bottom' of the grid.
            return {'x': 0, 'y': 1 + k // self.io_capacity}
        elif k < self.segment_start['top']:
            # Slot `k` maps to position along the ['right'] side of the grid.
            return {'x': int(1 + (k - self.segment_start['right']) //
                             self.io_capacity),
                    'y': int(self.extent['row'] + 1)}
        elif k < self.segment_start['left']:
            # Slot `k` maps to position along the top of the grid.
            return {'x': int(self.extent['column'] + 1),
                    'y': int(self.extent['row'] -
                             (k - self.segment_start['top']) //
                             self.io_capacity)}
        else:
            # Assume slot `k` maps to position along the left of the grid.
            return {'x': int(self.extent['column'] -
                             (k - self.segment_start['left']) //
                             self.io_capacity),
                    'y': 0}


class VPRAutoSlotKeyTo2dPosition(object):
    def __init__(self, io_count, logic_count, io_capacity=2):
        self.io_capacity = io_capacity
        self.io_count = io_count
        self.logic_count = logic_count
        self.extent = {}
        self.extent['row'] = math.ceil(math.sqrt(logic_count))
        self.extent['column'] = self.extent['row']
        if 2 * sum(self.extent.values()) * self.io_capacity < io_count:
            # The size determined based on the number of logic blocks does not
            # provide enough spots for the inputs/outputs along the perimeter.
            # Increase extents of the grid to fit IO.
            self.extent['row'] = math.ceil(math.sqrt(io_count + logic_count))
            self.extent['column'] = self.extent['row']

        if (io_count > 0):
            io_extent = self.extent
        else:
            io_extent = dict([(k, 0) for k, v in self.extent.iteritems()])

        self.io_s2p = VPRIOSlotKeyTo2dPosition(io_extent, self.io_capacity)
        self.logic_s2p = SlotKeyTo2dPosition(self.extent['row'],
                                             {'x': 1, 'y': 1})
        self.slot_count = {'io': len(self.io_s2p),
                           'logic': int(np.product(self.extent.values()))}

    def __len__(self):
        return sum(self.slot_count.values())

    def position0(self, position):
        return {'x': position['x'] - self.logic_s2p.offset['x'],
                'y': position['y'] - self.logic_s2p.offset['y']}

    def in_bounds(self, position):
        p = self.position0(position)
        return not (p['x'] < 0 or p['x'] >= self.extent['column'] or p['y'] < 0
                    or p['y'] >= self.extent['row'])

    def __getitem__(self, k):
        return self.get(k)

    def get0(self, k):
        return self.position0(self.get(k))

    def get(self, k):
        if k < len(self.io_s2p):
            return self.io_s2p[k]
        else:
            return self.logic_s2p[k - len(self.io_s2p)]


class MovePattern(object):
    def __init__(self, magnitude, shift=0):
        self.magnitude = magnitude
        self.double_magnitude = 2 * magnitude
        self.shift = shift

    def __getitem__(self, i):
        if self.magnitude == 0:
            return 0
        index = (i + 2 * self.double_magnitude -
                 ((self.shift + self.magnitude + 1) % self.double_magnitude))
        if (index % self.double_magnitude) < self.magnitude:
            return self.magnitude
        else:
            return -self.magnitude


class MovePattern2d(object):
    def __init__(self, magnitude, shift, extent):
        self.patterns = {'row': MovePattern(magnitude['row'], shift['row']),
                         'column': MovePattern(magnitude['column'],
                                               shift['column'])}
        self.extent = extent
        self.size = extent['row'] * extent['column']

    def column_i(self, i):
        return ((i // self.extent['row']) + self.extent['column'] *
                (i % self.extent['row']))

    def column(self, i):
        return self.patterns['column'][self.column_i(i)]

    def row(self, i):
        return self.patterns['row'][i]

    def __getitem__(self, i):
        return {'column': self.column(i), 'row': self.row(i)}

    def __len__(self):
        return self.size


class MovePatternInBounds(object):
    def __init__(self, extent, magnitude, shift=0):
        self.extent = extent
        self.pattern = MovePattern(magnitude, shift)

    def __getitem__(self, i):
        # We still need to use the offset-based position for computing the
        # target position.
        move = self.pattern[i]
        target = i + move
        if target < 0 or target >= self.extent:
            # If the displacement targets a location that is outside the
            # boundaries, set displacement to zero.
            return 0
        return int(move)

    def __len__(self):
        return self.extent


class MovePatternInBounds2d(object):
    def __init__(self, magnitude, shift, slot_key_to_position):
        self.s2p = slot_key_to_position
        self.pattern = MovePattern2d(magnitude, shift, self.s2p.extent)
        self.patterns = self.pattern.patterns

    def __getitem__(self, i):
        # Get zero-based position, since displacement patterns are indexed
        # starting at zero.
        position0 = self.s2p.get0(i)

        # We still need to use the offset-based position for computing the
        # target position.
        position = self.s2p.get(i)
        move = {'column': self.patterns['column'][position0['x']],
                'row': self.patterns['row'][position0['y']]}
        target = {'x': position['x'] + move['column'],
                  'y': position['y'] + move['row']}
        if not self.s2p.in_bounds(target):
            # If the displacement targets a location that is outside the
            # boundaries, set displacement to zero.
            return 0
        return int(move['column'] * self.s2p.extent['row'] + move['row'])

    def __len__(self):
        return len(self.pattern)


class VPRMovePattern(object):
    def __init__(self, io_magnitude, io_shift, logic_magnitude, logic_shift,
                 slot_key_to_position):
        self.s2p = slot_key_to_position
        self.io_slot_count = self.s2p.slot_count['io']
        self.io_pattern = MovePatternInBounds(self.io_slot_count,
                                              io_magnitude, io_shift)
        self.logic_pattern = MovePatternInBounds2d(logic_magnitude,
                                                   logic_shift, self.s2p)

    def __getitem__(self, i):
        if i < self.io_slot_count:
            return self.io_pattern[i]
        else:
            return self.logic_pattern[i]

    def __len__(self):
        return len(self.s2p)


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
    max_magnitude['row'] = min(vpr_s2p.extent['row'] - 1, max_magnitude['row'])
    max_magnitude['column'] = min(vpr_s2p.extent['column'] - 1,
                                  max_magnitude['column'])
    magnitude = dict([(k, np.random.randint(0, v + 1))
                      for k, v in max_magnitude.iteritems()])
    while (magnitude['row'] == 0) and (magnitude['column'] == 0):
        magnitude = dict([(k, np.random.randint(0, v + 1))
                          for k, v in max_magnitude.iteritems()])

    shift = {'row': 0, 'column': 0}

    if magnitude['row'] > 0:
        max_shift = 2 * magnitude['row'] - 1
        if vpr_s2p.extent['row'] <= 2 * magnitude['row']:
            max_shift = magnitude['row'] - 1
        shift['row'] = np.random.randint(max_shift + 1)

    if magnitude['column'] > 0:
        max_shift = 2 * magnitude['column'] - 1
        if vpr_s2p.extent['column'] <= 2 * magnitude['column']:
            max_shift = magnitude['column'] - 1
        shift['column'] = np.random.randint(max_shift + 1)
    return magnitude, shift


def random_vpr_pattern(vpr_s2p, max_logic_move=None, max_io_move=None):
    io_extent = vpr_s2p.slot_count['io']
    if max_io_move is None:
        max_io_move = io_extent - 1
    io_move, io_shift = random_pattern_params(max_io_move, io_extent)
    if max_logic_move is None:
        max_logic_move = {'row': vpr_s2p.extent['row'] - 1,
                          'column': vpr_s2p.extent['column'] - 1}
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
        self.slot_block_keys.fill(netlist.block_count)

        # Fill IO slots.
        self.slot_block_keys[:self.io_count] = [i for i, t in
                                                enumerate(netlist.block_types)
                                                if t in ('.input',
                                                         '.output')][:self
                                                                     .io_count]
        # Fill logic slots.
        logic_start = self.s2p.slot_count['io']
        logic_end = logic_start + self.logic_count
        self.slot_block_keys[logic_start:logic_end] = [i for i, t in
                                                       enumerate(netlist
                                                                 .block_types)
                                                       if t in ('.clb')]

        # Create reverse-mapping, from each block-key to the permutation slot
        # the block occupies.
        self.block_slot_keys = np.empty(netlist.block_count, dtype=np.uint32)
        self._sync_slot_block_keys_to_block_slot_keys()

        self.p_x = np.empty(netlist.block_count)
        self.p_y = np.empty(netlist.block_count)
        self.p_x_prime = np.empty(netlist.block_count)
        self.p_y_prime = np.empty(netlist.block_count)

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
        np.random.shuffle(self.slot_block_keys[:p.s2p.slot_count['io']])
        np.random.shuffle(self.slot_block_keys[p.s2p.slot_count['io']:])
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
        netlist = self.netlist
        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments.
        for i in xrange(netlist.block_count):
            position = self.s2p[self.block_slot_keys[i]]
            self.p_x[i] = position['x']
            self.p_y[i] = position['y']

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
        self.omega.data[:] = map(self.e_c.A.ravel().__getitem__, netlist.C.col)

        # $\vec{n_c}$ contains the total cost of all edges connected to node
        # $i$.
        self.n_c = self.omega.sum(axis=1)
        return self.theta

    def propose_moves(self, seed):
        netlist = self.netlist
        np.random.seed(seed)
        self.move_pattern = random_vpr_pattern(self.s2p)

        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments.
        for i in xrange(netlist.block_count):
            block_slot_key = self.block_slot_keys[i]
            position = self.s2p[block_slot_key +
                                self.move_pattern[block_slot_key]]
            self.p_x_prime[i] = position['x']
            self.p_y_prime[i] = position['y']

        self.block_slot_keys_prime = np.fromiter((k + self.move_pattern[k]
                                                  for k in
                                                  self.block_slot_keys),
                                                 dtype=np.int32)

    def evaluate_moves(self):
        for k, (i, j) in enumerate(itertools.izip(self.omega_prime.row,
                                                  self.omega_prime.col)):
            self.omega_prime.data[k] = 1.59 * (
                np.sqrt(np.square(self.e_x[0, j] - self.p_x[i] +
                                  self.p_x_prime[i]) - (self.e_x2[0, j] -
                                                        self.p_x[i] *
                                                        self.p_x[i] +
                                                        self.p_x_prime[i] *
                                                        self.p_x_prime[i]) *
                        self.r_inv[0, j] + 1) +
                np.sqrt(np.square(self.e_y[0, j] - self.p_y[i] +
                                  self.p_y_prime[i]) - (self.e_y2[0, j] -
                                                        self.p_y[i] *
                                                        self.p_y[i] +
                                                        self.p_y_prime[i] *
                                                        self.p_y_prime[i]) *
                        self.r_inv[0, j] + 1))

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
        self.block_slot_keys_prime[rejected_move_block_keys] = (
            self.block_slot_keys[rejected_move_block_keys])
        self.block_slot_keys, self.block_slot_keys_prime = (
            self.block_slot_keys_prime, self.block_slot_keys)
        self._sync_block_slot_keys_to_slot_block_keys()

    def run_iteration(self, seed, temperature):
        self.propose_moves(seed)
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
