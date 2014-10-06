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
import time
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from collections import OrderedDict

from cythrust.si_prefix import si_format
from cyplace_experiments.data import open_netlists_h5f
from .CAMIP import (evaluate_moves, VPRAutoSlotKeyTo2dPosition,
                    random_vpr_pattern, slot_moves, extract_positions,
                    cAnnealSchedule, get_std_dev, sort_netlist_keys,
                    sum_float_by_key, sum_xy_vectors, compute_block_group_keys,
                    minus_float, sum_permuted_float_by_key,
                    star_plus_2d, equal_count_uint32, sequence_int32,
                    permuted_nonmatch_inclusive_scan_int32, assess_groups,
                    copy_permuted_uint32, pack_io)

from cythrust.device_vector import DeviceVectorInt32, DeviceVectorFloat32
from device.CAMIP import (evaluate_moves as d_evaluate_moves,
                          minus_float as d_minus_float)


try:
    profile
except:
    profile = lambda (f): f


class VPRSchedule(object):
    def __init__(self, s2p, inner_num, netlist, placer=None):
        self.s2p = s2p
        self.inner_num = inner_num
        self.moves_per_temperature = inner_num * pow(netlist.block_count,
                                                     1.33333)
        rlim = min(*self.s2p.extent)
        if placer is not None:
            start_temperature = self.get_starting_temperature(placer)
        else:
            start_temperature = 0.
        self.anneal_schedule = cAnnealSchedule(rlim, start_temperature)

    def get_starting_temperature(self, placer, move_count=None):
        deltas_sum = 0.
        deltas_squared_sum = 0.
        total_moves = 0

        if move_count is None:
            move_count = placer.netlist.block_count

        while total_moves < move_count:
            placer.propose_moves(np.random.randint(100000))
            placer.evaluate_moves()
            non_zero_moves, rejected = placer.assess_groups(1000000)
            total_moves += non_zero_moves
            deltas = np.abs(placer.n_c_prime.ravel() - placer.n_c)
            deltas_sum += deltas.sum()
            deltas_squared_sum += (deltas * deltas).sum()
        deltas_stddev = get_std_dev(total_moves, deltas_squared_sum, deltas_sum
                                    / total_moves)
        return 20 * deltas_stddev

    def outer_iteration(self, placer):
        total_moves = 0
        rejected_moves = 0

        while total_moves < self.moves_per_temperature:
            max_logic_move = max(self.anneal_schedule.rlim, 1)
            non_zero_moves, rejected = placer.run_iteration(
                np.random.randint(1000000), self.anneal_schedule.temperature,
                max_logic_move=(max_logic_move, max_logic_move))
            total_moves += non_zero_moves
            rejected_moves += len(rejected)
        success_ratio = (total_moves - rejected_moves) / float(total_moves)
        self.anneal_schedule.update_state(success_ratio)
        return total_moves, rejected_moves

    def run(self, placer):
        states = []
        total_move_count = 0
        while (self.anneal_schedule.temperature > 0.00001 * placer.theta /
               placer.netlist.C.shape[1]):
            start = time.time()
            total_moves, rejected_moves = self.outer_iteration(placer)
            end = time.time()
            total_move_count += total_moves
            state = OrderedDict([('start', start), ('end', end),
                                 ('cost', placer.theta),
                                 ('temperature',
                                  self.anneal_schedule.temperature),
                                 ('success_ratio',
                                  self.anneal_schedule.success_ratio),
                                 ('radius_limit', self.anneal_schedule.rlim),
                                 ('total_iteration_count', total_move_count)])
            if not states:
                print '\n| ' + ' | '.join(state.keys()[2:]) + ' |'
                print '|' + '|'.join(['-' * (len(k) + 2)
                                      for k in state.keys()[2:]]) + '|'
            states.append(state)
            print '|' + '|'.join(['{0:>{1}s}'.format(si_format(v, 2),
                                                     len(k) + 2)
                                  for k, v in state.items()[2:]]) + '|'
        placer.finalize()
        print '\nRuntime: %.2fs' % (states[-1]['end'] - states[0]['start'])
        return states


class MatrixNetlist(object):
    def __init__(self, connections, block_types):
        CLOCK_PIN = 5
        self.connections = connections
        self.global_nets = (connections[connections['pin_key'] ==
                                        CLOCK_PIN]['net_key'].unique())
        global_block_mask = (connections['net_key'].isin(self.global_nets) &
                             (connections['pin_key'] == 0))
        self.global_blocks = (connections[global_block_mask]['block_key']
                              .unique().astype(dtype=np.int32))

        # Filter out connections that correspond to a global net.
        self.local_connections = connections[~connections['net_key']
                                             .isin(self.global_nets)]
        net_keys = self.local_connections['net_key'].copy().ravel()
        packed_keys = np.empty_like(net_keys)
        packed_keys[0] = 1
        packed_keys[1:] = (net_keys[1:] != net_keys[:-1])
        self.local_connections['block_type'] = \
            block_types[self.local_connections['block_key'].ravel()]
        self.local_connections['net_key'] = np.cumsum(packed_keys) - 1

        block_connections = self.local_connections.copy().sort(columns=
                                                               ['block_key'])
        block_keys = block_connections['block_key'].copy().ravel()
        packed_keys = np.empty_like(block_keys)
        packed_keys[0] = 1
        packed_keys[1:] = (block_keys[1:] != block_keys[:-1])
        block_connections['block_key'] = np.cumsum(packed_keys) - 1
        self.local_connections = block_connections.sort(columns=['net_key'])
        self.block_types = (self.local_connections
                            .drop_duplicates(cols='block_key')
                            .sort(['block_key',
                                   'block_type'])['block_type'].copy())

        b_types = OrderedDict([(v, i) for i, v in enumerate(('.clb', '.input',
                                                             '.output'))])
        keys = np.sort(np.array(map(b_types.__getitem__, self.block_types),
                                dtype=np.int32))
        counts = np.ones_like(keys, dtype=np.float32)
        N = sum_float_by_key(keys, counts, keys, counts)
        self.block_type_counts = OrderedDict(zip(b_types.keys(),
                                                 counts[:N].astype(int)))
        self.block_count = counts[:N].sum()
        self.non_global_block_count = self.block_count - self.global_nets.size

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


class CAMIP(object):
    def __init__(self, netlist, io_capacity=3):
        self.netlist = netlist
        self.io_capacity = io_capacity

        self.io_count = np.sum(netlist.block_type_counts[t]
                               for t in ('.input', '.output'))
        self.logic_count = netlist.block_type_counts['.clb']
        self.s2p = VPRAutoSlotKeyTo2dPosition(self.io_count, self.logic_count,
                                              io_capacity)
        self.slot_block_keys = np.empty(self.s2p.total_slot_count,
                                        dtype=np.uint32)
        self.slot_block_keys[:] = netlist.block_count

        # Fill IO slots.
        self.slot_block_keys[:self.io_count] = np.where(netlist.block_types
                                                        .isin(('.input',
                                                               '.output')))[0]

        # Fill logic slots.
        logic_start = self.s2p.io_slot_count
        logic_end = logic_start + self.logic_count
        self.slot_block_keys[logic_start:logic_end] = np.where(
            netlist.block_types == '.clb')[0]

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

        self._e_x = np.empty(self.X.shape[1], dtype=np.float32)
        self._e_x2 = np.empty_like(self._e_x)
        self._e_y = np.empty_like(self._e_x)
        self._e_y2 = np.empty_like(self._e_x)
        self._e_c = np.empty_like(self._e_x)

        self.omega = netlist.C._with_data(np.empty_like(netlist.C.data,
                                                        dtype=np.float32),
                                          copy=False)
        self.omega_prime = netlist.C._with_data(np.ones_like(netlist.C.data,
                                                             dtype=np
                                                             .float32),
                                                copy=True)
        self._block_keys = np.empty_like(netlist.C.row)
        self.net_count = sum_float_by_key(self.omega_prime.col,
                                          self.omega_prime.data,
                                          self._block_keys,
                                          self.omega_prime.data)
        sort_netlist_keys(self.omega.row, self.omega.col)
        self.omega_prime.row[:] = self.omega.row
        self.omega_prime.col[:] = self.omega.col

        self.e_x = self._e_x[:self.net_count]
        self.e_x2 = self._e_x2[:self.net_count]
        self.e_y = self._e_y[:self.net_count]
        self.e_y2 = self._e_y2[:self.net_count]
        self.e_c = self._e_c[:self.net_count]

        self.n_c_prime = self.omega_prime.data[:netlist.block_count,
                                               np.newaxis]
        self.r_inv = np.reciprocal(netlist.r.A.ravel().astype(np.float32))
        self.block_group_keys = np.empty(netlist.block_count, dtype=np.int32)
        self._group_block_keys = np.empty(self.block_group_keys.size,
                                          dtype=self.block_group_keys.dtype)
        self._delta_s = np.empty(netlist.block_count, dtype=np.float32)
        self._n_c = np.empty_like(netlist.C.row, dtype=np.float32)
        self.n_c = self._n_c[:netlist.block_count]
        self.delta_n = np.empty(netlist.block_count, dtype=np.float32)
        self._packed_block_group_keys = np.empty_like(self._group_block_keys)
        self._assess_urands = np.empty_like(self._delta_s)
        self._rejected_block_keys = np.empty_like(self.block_group_keys)

    def shuffle_placement(self):
        '''
        Shuffle placement permutation.

        The shuffle is aware of IO and logic slots in the placement, and will
        keep IO and logic within the corresponding areas of the permutation.
        '''
        np.random.shuffle(self.slot_block_keys[:self.s2p.io_slot_count])
        np.random.shuffle(self.slot_block_keys[self.s2p.io_slot_count:])
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

    @profile
    def evaluate_placement(self):
        '''
        Compute the cost of:

         - Each net _(`self.e_c`)_.
         - The complete placement _(`self.theta`)_.
        '''
        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments.
        # Thrust `transform`.
        extract_positions(self.block_slot_keys, self.p_x, self.p_y, self.s2p)

        # Star+ vectors
        # Thrust `reduce_by_key`.
        sum_xy_vectors(self.X.row, self.X.col, self.p_x, self.p_y, self._e_x,
                       self._e_x2, self._e_y, self._e_y2, self._block_keys)

        # `theta`: $\theta =$ total placement cost
        # Thrust `transform`.
        self.theta = star_plus_2d(self.e_x, self.e_x2, self.e_y, self.e_y2,
                                  self.r_inv, 1.59, self.e_c)

        # $\vec{n_c}$ contains the total cost of all edges connected to node
        # $i$.
        # Thrust `reduce_by_key`.
        N = sum_permuted_float_by_key(self.omega.row, self.e_c.ravel(),
                                      self.omega.col, self._block_keys,
                                      self._n_c)
        return self.theta

    @profile
    def propose_moves(self, seed, max_io_move=None, max_logic_move=None):
        '''
        Based on the provided seed:
            - Generate a random move pattern _(constant time)_.
            - Compute the new permutation slot for each block in the placement,
              based on the generated moves pattern.
            - Compute the `(x, y)` corresponding to the proposed slot for each
              block.


        Notes
        =====

         - All operations that are not constant-time are implemented using
           Thrust [`transform`][1] operations _(i.e., [map of sequence][2])_.

        [1]: http://thrust.github.io/doc/group__transformations.html
        [2]: http://www.sciencedirect.com/science/article/pii/B9780124159938000049
        '''
        # TODO: Use C++ random number generator.
        np.random.seed(seed)
        # TODO: Implement pure C++ random VPR pattern generator.
        self.move_pattern = random_vpr_pattern(self.s2p,
                                               max_io_move=max_io_move,
                                               max_logic_move=max_logic_move)
        # Thrust `transform`.
        slot_moves(self.block_slot_keys, self.block_slot_keys_prime,
                   self.move_pattern)

        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments, using Thrust `transform`.
        extract_positions(self.block_slot_keys_prime, self.p_x_prime,
                          self.p_y_prime, self.s2p)

    @profile
    def evaluate_moves(self):
        '''
         - Compute the total cost per block, based on the current set of
           proposed moves.
         - For each block, compute the difference in cost between the current
           position and the newly proposed position.


        Notes
        =====

         - All operations are implemented using one of the following Thrust
           operations:

          - [`reduce_by_key`][1] _(i.e., [category reduction][2])_.
          - [`transform`][3] _(i.e., [map of sequence][4])_.

        [1]: http://thrust.github.io/doc/group__reductions.html#ga633d78d4cb2650624ec354c9abd0c97f
        [2]: http://www.sciencedirect.com/science/article/pii/B9780124159938000037#s0175
        [3]: http://thrust.github.io/doc/group__transformations.html
        [4]: http://www.sciencedirect.com/science/article/pii/B9780124159938000049
        '''
        # Thrust `reduce_by_key` over a `transform` iterator.
        #evaluate_moves(self.omega_prime.row,
                       #self.omega_prime.col,
                       #self.p_x, self.p_x_prime, self.e_x, self.e_x2,
                       #self.p_y, self.p_y_prime, self.e_y, self.e_y2,
                       #self.r_inv, 1.59, self._block_keys,
                       #self.omega_prime.data)
        row = DeviceVectorInt32(self.omega_prime.col.size); row[:] = self.omega_prime.row
        col = DeviceVectorInt32(self.omega_prime.col.size); col[:] = self.omega_prime.col
        p_x = DeviceVectorInt32(self.p_x.size); p_x[:] = self.p_x
        p_x_prime = DeviceVectorInt32(self.p_x_prime.size); p_x_prime[:] = self.p_x_prime
        e_x = DeviceVectorFloat32(self.e_x.size); e_x[:] = self.e_x
        e_x2 = DeviceVectorFloat32(self.e_x2.size); e_x2[:] = self.e_x2
        p_y = DeviceVectorInt32(self.p_y.size); p_y[:] = self.p_y
        p_y_prime = DeviceVectorInt32(self.p_y_prime.size); p_y_prime[:] = self.p_y_prime
        e_y = DeviceVectorFloat32(self.e_y.size); e_y[:] = self.e_y
        e_y2 = DeviceVectorFloat32(self.e_y2.size); e_y2[:] = self.e_y2
        r_inv = DeviceVectorFloat32(self.r_inv.size); r_inv[:] = self.r_inv
        _block_keys = DeviceVectorInt32(self._block_keys.size)
        n_c_prime = DeviceVectorFloat32(self.omega_prime.data.size)

        d_evaluate_moves(row, col, p_x, p_x_prime, e_x, e_x2, p_y, p_y_prime,
                         e_y, e_y2, r_inv, 1.59, _block_keys, n_c_prime)
        self.omega_prime.data[:self.n_c_prime.size] = n_c_prime[:self.n_c_prime.size]

        # Compute move-deltas using a Thrust `reduce_by_key` call over a
        # `transform` iterator.
        n_c = DeviceVectorFloat32(self.n_c.size); n_c[:] = self.n_c
        delta_n = DeviceVectorFloat32(self.delta_n.size); delta_n[:] = self.delta_n
        #minus_float(self.n_c_prime.ravel(), self.n_c, self.delta_n)
        d_minus_float(n_c_prime, n_c, delta_n)
        self.delta_n[:] = delta_n[:]

    @profile
    def assess_groups(self, temperature):
        '''
         - For each block, compute the key of the group of
           concurrent-associated moves the block belongs to.
          - For a swap, which corresponds to a group of two
            concurrent-associated moves, the group key is equal to the minimum
            corresponding permutation slot key.
         - For each group of concurrent-associated moves, compute the
           difference in cost due to applying all moves in the group.
         - For each group of concurrent-associated moves, assess if _all_ moves
           in the group should be _accepted_ or _rejected_.

        Notes
        =====

         - All operations are implemented using one of the following Thrust
         operations:

          - [`reduce_by_key`][1] _(i.e., [category reduction][2])_.
          - [`transform`][3] _(i.e., [map][4])_.
          - [`sequence`][5] _(i.e., [map][4] to store a range of values)_.
          - [`inclusive_scan`][6] _(i.e., [fused map and prefix sum][7])_.
          - [`copy_if`][8] _(i.e., [fused map and pack][9])_.

        [1]: http://thrust.github.io/doc/group__reductions.html#ga633d78d4cb2650624ec354c9abd0c97f
        [2]: http://www.sciencedirect.com/science/article/pii/B9780124159938000037#s0175
        [3]: http://thrust.github.io/doc/group__transformations.html
        [4]: http://www.sciencedirect.com/science/article/pii/B9780124159938000049
        [5]: http://thrust.github.io/doc/sequence_8h.html
        [6]: http://thrust.github.io/doc/group__prefixsums.html#ga7109170b96a48fab736e52b75f423464
        [7]: http://www.sciencedirect.com/science/article/pii/B9780124159938000050#s0155
        [8]: http://thrust.github.io/doc/group__stream__compaction.html#ga36d9d6ed8e17b442c1fd8dc40bd515d5
        [9]: http://www.sciencedirect.com/science/article/pii/B9780124159938000062#s0065
        '''
        # Thrust `transform`.
        compute_block_group_keys(self.block_slot_keys,
                                 self.block_slot_keys_prime,
                                 self.block_group_keys,
                                 self.s2p.total_slot_count)

        # Thrust `reduce`.
        unmoved_count = equal_count_uint32(self.block_slot_keys,
                                           self.block_slot_keys_prime)

        # ## Packed block group keys ##
        #
        # Each block that has been assigned a non-zero move belongs to exactly
        # one group of associated-moves.  For each group of associated-moves,
        # there is a corresponding group key.  Given the index of a block in
        # the list of blocks belonging to groups of associated moves,

        # Thrust `sequence`.
        sequence_int32(self._group_block_keys)

        # Thrust `sort_by_key`.
        sort_netlist_keys(self.block_group_keys.copy(), self._group_block_keys)

        group_block_keys = self._group_block_keys[:-unmoved_count]
        if group_block_keys.size == 0:
            self.delta_s = self._delta_s[:0]
            return 0, np.empty(0)
        packed_block_group_keys = (self._packed_block_group_keys
                                   [:group_block_keys.size])

        # Thrust `inclusive_scan`.
        permuted_nonmatch_inclusive_scan_int32(self.block_group_keys,
                                               group_block_keys,
                                               packed_block_group_keys)

        # Thrust `reduce_by_key` over a `permutation` iterator.
        N = sum_permuted_float_by_key(packed_block_group_keys, self.delta_n,
                                      group_block_keys, self._block_keys,
                                      self._delta_s)
        self.delta_s = self._delta_s[:N]

        # Thrust `copy_if`.
        N = assess_groups(temperature, group_block_keys,
                          packed_block_group_keys, self.delta_s,
                          self._rejected_block_keys)
        rejected_block_keys = self._rejected_block_keys[:N]

        #      (moves evaluated)                , (moves rejected)
        return (self.block_slot_keys.size - unmoved_count), rejected_block_keys

    @profile
    def apply_groups(self, rejected_move_block_keys):
        '''
         - Update placement according to accepted moves.

        Notes
        =====

         - All operations are implemented using one of the following Thrust
         operations:

          - [`transform`][1] _(i.e., [map][2])_.

        [1]: http://thrust.github.io/doc/group__transformations.html
        [2]: http://www.sciencedirect.com/science/article/pii/B9780124159938000049
        '''
        if len(rejected_move_block_keys) == 0:
            return

        # Thrust `transform`.
        copy_permuted_uint32(self.block_slot_keys, self.block_slot_keys_prime,
                             rejected_move_block_keys)
        self.block_slot_keys, self.block_slot_keys_prime = (
            self.block_slot_keys_prime, self.block_slot_keys)

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

    def finalize(self):
        # Copy the final position of each block to the slot-to-block-key
        # mapping.
        self._sync_block_slot_keys_to_slot_block_keys()

        # To make output compatible with VPR, we must pack blocks in IO tiles
        # to fill IO tile-slots contiguously.
        pack_io(self.slot_block_keys[:self.s2p.io_slot_count], self.io_capacity)


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
