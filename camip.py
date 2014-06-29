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
        self.net_ones = np.matrix([np.ones(self.C.shape[0])], dtype=np.uint8)
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
    io_count = (netlist.block_type_counts['.input'] +
                netlist.block_type_counts['.output'])
    logic_count = netlist.block_type_counts['.clb']
    s2p = VPRAutoSlotKeyTo2dPosition(io_count, logic_count, 2)
    slot_block_keys = np.empty(len(s2p), dtype=np.uint32)
    slot_block_keys.fill(netlist.block_count)

    # Fill IO slots.
    slot_block_keys[:io_count] = [i for i, t in enumerate(netlist.block_types)
                                  if t in ('.input', '.output')][:io_count]
    # Fill logic slots.
    logic_start = s2p.slot_count['io']
    logic_end = logic_start + logic_count
    slot_block_keys[logic_start:logic_end] = [i for i, t in
                                              enumerate(netlist.block_types)
                                              if t in ('.clb')]

    # Create reverse-mapping, from each block-key to the permutation slot the
    # block occupies.
    block_slot_keys = np.empty(netlist.block_count, dtype=np.uint32)
    occupied = np.where(slot_block_keys < netlist.block_count)
    block_slot_keys[slot_block_keys[occupied]] = occupied[0]
    del occupied

    p_x = np.empty(netlist.block_count)
    p_y = np.empty(netlist.block_count)
    p_x_prime = np.empty(netlist.block_count)
    p_y_prime = np.empty(netlist.block_count)

    Y = netlist.C._with_data(np.empty_like(netlist.C.data, dtype=np.float32),
                             copy=False)
    X = netlist.C._with_data(np.empty_like(netlist.C.data, dtype=np.float32),
                             copy=False)
    omega = netlist.C._with_data(np.empty_like(netlist.C.data,
                                               dtype=np.float32), copy=False)
    omega_prime = netlist.C._with_data(np.empty_like(netlist.C.data,
                                       dtype=np.float32), copy=False)
    r_inv = np.reciprocal(netlist.r.astype(np.float32))

    # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    # slot assignments.
    for i in xrange(netlist.block_count):
        position = s2p[block_slot_keys[i]]
        p_x[i] = position['x']
        p_y[i] = position['y']

    X.data[:] = p_x[X.row]
    Y.data[:] = p_y[Y.row]

    # Star+ vectors
    e_x = X.sum(axis=0)
    e_x2 = X.multiply(X).sum(axis=0)
    e_y = Y.sum(axis=0)
    e_y2 = Y.multiply(Y).sum(axis=0)
    e_c = 1.59 * (np.sqrt(np.square(e_x) - e_x2.A * r_inv.A + 1) +
                  np.sqrt(np.square(e_y) - e_y2.A * r_inv.A + 1))

    # `theta`: $\theta =$ total placement cost
    theta = e_c.sum()

    # `omega`: $\Omega \in \mathbb{M}_{mn}$, $\omega_{ij} = e_cj$.
    omega.data[:] = map(e_c.A.ravel().__getitem__, netlist.C.col)

    # $\vec{n_c}$ contains the total cost of all edges connected to node $i$.
    n_c = omega.sum(axis=1)

    np.random.seed(42)
    move_pattern = random_vpr_pattern(s2p)

    # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    # slot assignments.
    for i in xrange(netlist.block_count):
        block_slot_key = block_slot_keys[i]
        position = s2p[block_slot_key + move_pattern[block_slot_key]]
        p_x_prime[i] = position['x']
        p_y_prime[i] = position['y']

    for k, (i, j) in enumerate(itertools.izip(omega_prime.row,
                                              omega_prime.col)):
        omega_prime.data[k] = 1.59 * (np.sqrt(np.square(e_x[0, j] - p_x[i] +
                                                        p_x_prime[i]) -
                                              (e_x2[0, j] - p_x[i] * p_x[i] +
                                               p_x_prime[i] * p_x_prime[i]) *
                                              r_inv[0, j] + 1) +
                                      np.sqrt(np.square(e_y[0, j] - p_y[i] +
                                                        p_y_prime[i]) -
                                              (e_y2[0, j] - p_y[i] * p_y[i] +
                                               p_y_prime[i] * p_y_prime[i]) *
                                              r_inv[0, j] + 1))
    n_c_prime = omega_prime.sum(axis=1)
    delta_n = n_c_prime - n_c

    # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    # slot assignments.
    for i in xrange(netlist.block_count):
        block_slot_key = block_slot_keys[i]
        position = s2p[block_slot_key + move_pattern[block_slot_key]]
        p_x_prime[i] = position['x']
        p_y_prime[i] = position['y']

    block_slot_keys_prime = np.fromiter((k + move_pattern[k]
                                         for k in block_slot_keys),
                                        dtype=np.int32)
    total_slot_count = len(s2p)
    block_group_keys = np.fromiter((min(k1, k2) if k1 != k2
                                    else total_slot_count
                                    for k1, k2 in
                                    itertools.izip(block_slot_keys,
                                                   block_slot_keys_prime)),
                                   dtype=np.int32)
    moved_mask = (block_slot_keys != block_slot_keys_prime)
    unmoved_count = moved_mask.size - moved_mask.sum()

    group_block_keys = np.argsort(block_group_keys)[:-unmoved_count]
    packed_group_segments = np.empty_like(group_block_keys)
    packed_group_segments[0] = 1
    packed_group_segments[1:] = (block_group_keys[group_block_keys][1:] !=
                                 block_group_keys[group_block_keys][:-1])
    packed_block_group_keys = np.cumsum(packed_group_segments) - 1
    S = sparse.coo_matrix((delta_n[group_block_keys].A.ravel(),
                           (group_block_keys, packed_block_group_keys)))

    delta_s = S.sum(axis=0).A.ravel()

    assess_urands = np.random.rand(len(delta_s))
    temperature = 0.1
    a = ((delta_s <= 0) | (assess_urands < np.exp(-delta_s / temperature)))
    packed_group_keys = (block_group_keys[group_block_keys]
                         [np.where(packed_group_segments)])
    rejected_block_keys = S.row[~a[packed_block_group_keys]]
    block_slot_keys_prime[rejected_block_keys] = (block_slot_keys
                                                  [rejected_block_keys])
    slot_block_keys[:] = netlist.block_count
    slot_block_keys[block_slot_keys_prime] = xrange(netlist.block_count)
    block_slot_keys, block_slot_keys_prime = (block_slot_keys_prime,
                                              block_slot_keys)

    np.set_printoptions(precision=2, linewidth=250)
