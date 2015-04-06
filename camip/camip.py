# coding: utf-8
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
from fpga_netlist.connections_table import (INPUT_BLOCK, OUTPUT_BLOCK,
                                            LOGIC_BLOCK, CONNECTION_CLOCK,
                                            CONNECTION_CLOCK_DRIVER)
from .CAMIP import (VPRAutoSlotKeyTo2dPosition, random_vpr_pattern,
                    cAnnealSchedule, get_std_dev, pack_io,
                    # Only used in class constructors _(not in main methods)_.
                    sort_netlist_keys, sum_float_by_key)

from cythrust.device_vector import (DeviceVectorInt32, DeviceVectorFloat32,
                                    DeviceVectorUint32)
import cythrust.device_vector as dv
from device.CAMIP import (evaluate_moves as d_evaluate_moves,
                          minus_float as d_minus_float,
                          slot_moves as d_slot_moves,
                          extract_positions as d_extract_positions,
                          copy_permuted_uint32 as d_copy_permuted_uint32,
                          sum_xy_vectors as d_sum_xy_vectors,
                          star_plus_2d as d_star_plus_2d,
                          sum_permuted_float_by_key as
                          d_sum_permuted_float_by_key,
                          compute_block_group_keys as
                          d_compute_block_group_keys,
                          equal_count_uint32 as d_equal_count_uint32,
                          sequence_int32 as d_sequence_int32,
                          sort_netlist_keys as d_sort_netlist_keys,
                          permuted_nonmatch_inclusive_scan_int32 as
                          d_permuted_nonmatch_inclusive_scan_int32,
                          assess_groups as d_assess_groups,
                          sum_float_by_key as d_sum_float_by_key,
                          copy_int32 as d_copy_int32,
                          permuted_fill_float32 as d_permuted_fill_float32,
                          permuted_fill_int32 as d_permuted_fill_int32,
                          look_up_delay as d_look_up_delay)


try:
    profile
except:
    profile = lambda (f): f


class VPRSchedule(object):
    def __init__(self, s2p, inner_num, block_count, placer=None,
                 draw_enabled=False):
        self.s2p = s2p
        self.inner_num = inner_num
        self.moves_per_temperature = inner_num * pow(block_count, 1.33333)
        rlim = min(*self.s2p.extent)
        if placer is not None:
            start_temperature = self.get_starting_temperature(placer)
        else:
            start_temperature = 0.
        self.anneal_schedule = cAnnealSchedule(rlim, start_temperature)
        self.draw_enabled = draw_enabled

    def get_starting_temperature(self, placer, move_count=None):
        deltas_sum = 0.
        deltas_squared_sum = 0.
        total_moves = 0

        if move_count is None:
            move_count = placer.block_count
        block_count = int(placer.block_count)

        while total_moves < move_count:
            placer.propose_moves(np.random.randint(100000))
            placer.evaluate_moves()
            non_zero_moves, rejected = placer.assess_groups(1000000)
            total_moves += non_zero_moves
            deltas = np.abs(placer.omega_prime.data[:block_count] -
                            placer._n_c[:block_count])
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
            if self.draw_enabled and hasattr(placer, 'draw'):
                placer.draw()
            total_moves += non_zero_moves
            rejected_moves += rejected

        success_ratio = (total_moves - rejected_moves) / float(total_moves)
        self.anneal_schedule.update_state(success_ratio)
        placer.update_state(max_logic_move)
        return total_moves, rejected_moves

    def run(self, placer):
        states = []
        #while (self.anneal_schedule.temperature > 0.00001 * placer.theta /
               #placer.net_count):
        while (not placer.exit_criteria(self.anneal_schedule.temperature)):
            start = time.time()
            evaluated_moves, rejected_moves = self.outer_iteration(placer)
            end = time.time()
            extra_state = placer.get_state()
            state = OrderedDict([('start', start), ('end', end),
                                 ('cost', placer.theta),
                                 ('temperature',
                                  self.anneal_schedule.temperature),
                                 ('success_ratio',
                                  self.anneal_schedule.success_ratio),
                                 ('radius_limit', self.anneal_schedule.rlim),
                                 ('evaluated_move_count', evaluated_moves)])
            state.update(extra_state)
            if not states:
                print '\n| ' + ' | '.join(state.keys()[2:]) + ' |'
                print '|' + '|'.join(['-' * (len(k) + 2)
                                      for k in state.keys()[2:]]) + '|'
            states.append(state)
            print '|' + '|'.join(['{0:>{1}s}'.format(si_format(v, 2),
                                                     len(k) + 2)
                                  for k, v in state.items()[2:]]) + '|'
        placer.finalize()
        return states


class DeviceSparseMatrix(object):
    def __init__(self, row, col, data=None):
        self.row = DeviceVectorInt32(row.size)
        self.col = DeviceVectorInt32(col.size)
        self.row[:] = row[:]
        self.col[:] = col[:]
        if data is None:
            self.data = None
        else:
            if data.dtype == np.float32:
                self.data = DeviceVectorFloat32(data.size)
            elif data.dtype == np.int32:
                self.data = DeviceVectorInt32(data.size)
            elif data.dtype == np.uint32:
                self.data = DeviceVectorUint32(data.size)
            self.data[:] = data[:]


class CAMIP(object):
    def __init__(self, connections_table, io_capacity=3, include_clock=False):
        self._connections_table = connections_table

        # __NB__ The code now allows for _any_ connections to be filtered out.
        # Only blocks and nets referenced by the selected connections will
        # affect _wire-length_ optimization.
        # For example, a random mask of some the connections could be used,
        # like so:
        #
        #     exclude_count = int(self._connections_table.connections.shape[0] / 30.)
        #     include_count = int(self._connections_table.connections.shape[0] -
                                 #exclude_count)
        #     mask = np.array(include_count * [True] + exclude_count * [False])
        #     np.random.shuffle(mask)

        # Only optimize using non-global connections/nets _(i.e., filter out
        # any global net connections)_.
        if include_clock:
            mask = np.ones(connections_table.connections.shape[0], dtype=np.bool)
        else:
            mask = ~connections_table.connections.type.isin([CONNECTION_CLOCK,
                                                             CONNECTION_CLOCK_DRIVER])
        self.connections = self._connections_table.filter(mask)
        self.io_capacity = io_capacity

        self.io_count = connections_table.io_count
        self.logic_count = connections_table.logic_count
        self.block_count = self.io_count + self.logic_count
        #self.net_count = connections_table.net_count
        self.net_count = self.connections.net_key.unique().size

        self.s2p = VPRAutoSlotKeyTo2dPosition(self.io_count, self.logic_count,
                                              io_capacity)
        self.slot_block_keys = DeviceVectorUint32(self.s2p.total_slot_count)
        self.slot_block_keys[:] = self.block_count

        # Fill IO slots.
        self.slot_block_keys[:self.io_count] = \
            connections_table.io_block_keys()

        # Fill logic slots.
        logic_start = self.s2p.io_slot_count
        logic_end = logic_start + self.logic_count
        self.slot_block_keys[logic_start:logic_end] = \
            connections_table.logic_block_keys()

        # Create reverse-mapping, from each block-key to the permutation slot
        # the block occupies.
        self.block_slot_keys = DeviceVectorUint32(self.block_count)
        self.block_slot_moves = DeviceVectorInt32(self.block_count)
        self.block_slot_keys_prime = DeviceVectorUint32(self.block_count)
        self._sync_slot_block_keys_to_block_slot_keys()

        self.p_x = DeviceVectorInt32(self.block_count)
        self.p_y = DeviceVectorInt32(self.block_count)
        self.p_x_prime = DeviceVectorInt32(self.block_count)
        self.p_y_prime = DeviceVectorInt32(self.block_count)

        self.X = DeviceSparseMatrix(
            self.connections.sink_key,
            self.connections.net_key)

        self._e_x = dv.from_array(np.zeros(self.X.col.size, dtype=np.float32))
        self._e_x2 = dv.from_array(np.zeros(self.X.col.size, dtype=np.float32))
        self._e_y = dv.from_array(np.zeros(self.X.col.size, dtype=np.float32))
        self._e_y2 = dv.from_array(np.zeros(self.X.col.size, dtype=np.float32))
        self._e_c = dv.from_array(np.zeros(self.X.col.size, dtype=np.float32))

        self.omega = DeviceSparseMatrix(
            self.connections.sink_key,
            self.connections.net_key)
        self.omega_prime = DeviceSparseMatrix(
            self.connections.sink_key,
            self.connections.net_key,
            np.ones(len(self.connections), dtype=np.float32))
        self._block_keys = DeviceVectorInt32(len(self.connections))

        d_sort_netlist_keys(self.X.col, self.X.row)
        d_sort_netlist_keys(self.omega.row, self.omega.col)

        self.omega_prime.row[:] = self.omega.row[:]
        self.omega_prime.col[:] = self.omega.col[:]

        self.r_inv = dv.from_array(np.ones(self.net_count, dtype=np.float32))
        self.non_global_net_count = d_sum_float_by_key(self.X.col,
                                                       self.omega_prime.data,
                                                       self._block_keys,
                                                       self.omega_prime.data)
        r_inv = self.r_inv[:]
        r_inv[self._block_keys[:self.non_global_net_count]] = np.reciprocal(
            self.omega_prime.data[:self.non_global_net_count])
        self.r_inv[:] = r_inv
        self.block_group_keys = DeviceVectorInt32(self.block_count)
        self.block_group_keys_sorted = DeviceVectorInt32(self.block_count)
        self._group_block_keys = DeviceVectorInt32(self.block_group_keys.size)
        self._delta_s = dv.from_array(np.zeros(self.block_count,
                                               dtype=np.float32))
        self._n_c = dv.from_array(np.zeros(len(self.connections),
                                           dtype=np.float32))
        self.delta_n = dv.from_array(np.zeros(len(self.connections),
                                              dtype=np.float32))
        self._packed_block_group_keys = DeviceVectorInt32(self
                                                          ._group_block_keys
                                                          .size)
        self._rejected_block_keys = DeviceVectorInt32(self.block_group_keys
                                                      .size)
        self.omega_prime.data[:] = 0

    def get_net_elements(self):
        return pd.DataFrame(dict([(k, getattr(self, '_e_%s' %
                                              k)[:self.net_count]) for k in
                                  ('x', 'x2', 'y', 'y2')]))

    def connection_data(self):
        data = pd.DataFrame({'sink_key': self.omega_prime.row[:], 'net_key': self.omega_prime.col[:]})
        data['p_x'] = self.p_x[:][data['sink_key']]
        data['p_y'] = self.p_y[:][data['sink_key']]
        data['p_x2'] = data['p_x'] ** 2
        data['p_y2'] = data['p_y'] ** 2
        data.head()
        data['p_x_prime'] = self.p_x_prime[:][data['sink_key']]
        data['p_y_prime'] = self.p_y_prime[:][data['sink_key']]
        data['p_x2_prime'] = data['p_x_prime'] ** 2
        data['p_y2_prime'] = data['p_y_prime'] ** 2
        data['r_inv'] = self.r_inv[:][data['net_key']]
        data.sort('net_key', inplace=True)
        return data

    def star_plus_elements(self, data):
        elements = data.groupby('net_key', as_index=False).apply(
            lambda u: pd.DataFrame(OrderedDict(
                [('%s%s' % (k, p), np.sum(getattr(u, 'p_%s%s' % (k, p))))
                 for p in ('', '_prime') for k in ('x', 'x2', 'y', 'y2')]),
                index=[u.net_key.iloc[0]]))
        return elements

    def star_plus_delta(self, data):
        star_plus = data.groupby('net_key').apply(lambda u: 1.59 *
                                                  (np.sqrt(np.sum(u.p_x2) -
                                                           np.sum(u.p_x) ** 2 *
                                                           u.r_inv.iloc[0] + 1)
                                                   + np.sqrt(np.sum(u.p_y2) -
                                                             np.sum(u.p_y) ** 2
                                                             * u.r_inv.iloc[0]
                                                             + 1)))
        star_plus_prime = data.groupby('net_key').apply(lambda u: 1.59 *
                                                        (np.sqrt(np.sum(u.p_x2_prime)
                                                                 -
                                                                 np.sum(u.p_x_prime)
                                                                 ** 2 *
                                                                 u.r_inv.iloc[0]
                                                                 + 1) +
                                                         np.sqrt(np.sum(u.p_y2_prime)
                                                                 -
                                                                 np.sum(u.p_y_prime)
                                                                 ** 2 *
                                                                 u.r_inv.iloc[0]
                                                                 + 1)))
        return star_plus, star_plus_prime

    def exit_criteria(self, temperature):
        return temperature < 0.00001 * self.theta / self.net_count

    def update_state(self, maximum_move_distance):
        pass

    def shuffle_placement(self):
        '''
        Shuffle placement permutation.

        The shuffle is aware of IO and logic slots in the placement, and will
        keep IO and logic within the corresponding areas of the permutation.
        '''
        slot_block_keys = self.slot_block_keys[:]
        np.random.shuffle(slot_block_keys[:self.s2p.io_slot_count])
        np.random.shuffle(slot_block_keys[self.s2p.io_slot_count:])
        self.slot_block_keys[:] = slot_block_keys[:]
        self._sync_slot_block_keys_to_block_slot_keys()

    def _sync_block_slot_keys_to_slot_block_keys(self):
        '''
        Update `slot_block_keys` based on `block_slot_keys`.

        Useful when updating the permutation slot contents directly _(e.g.,
        shuffling the contents of the permutation slots)_.
        '''
        slot_block_keys = self.slot_block_keys[:]
        slot_block_keys[:] = self.block_count
        slot_block_keys[self.block_slot_keys[:]] = xrange(self.block_count)
        self.slot_block_keys[:] = slot_block_keys[:]

    def _sync_slot_block_keys_to_block_slot_keys(self):
        '''
        Update `block_slot_keys` based on `slot_block_keys`.
        '''
        occupied = np.where(self.slot_block_keys[:] < self.block_count)
        block_slot_keys = self.block_slot_keys[:]
        slot_block_keys = self.slot_block_keys[:]
        block_slot_keys[slot_block_keys[occupied]] = occupied[0]
        self.block_slot_keys[:] = block_slot_keys[:]

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
        d_extract_positions(self.block_slot_keys, self.p_x, self.p_y, self.s2p)

        # Star+ vectors
        # Thrust `reduce_by_key`.
        d_sum_xy_vectors(self.X.row, self.X.col, self.p_x, self.p_y, self._e_x,
                         self._e_x2, self._e_y, self._e_y2, self._block_keys)

        # `theta`: $\theta =$ total placement cost
        # Thrust `transform`.
        self.theta = d_star_plus_2d(self._e_x, self._e_x2, self._e_y,
                                    self._e_y2, self.r_inv, 1.59, self._e_c)

        # $\vec{n_c}$ contains the total cost of all edges connected to node
        # $i$.
        # Thrust `reduce_by_key`.
        N = d_sum_permuted_float_by_key(self.omega.row, self._e_c,
                                        self.omega.col, self._block_keys,
                                        self._n_c, self.omega.col.size)
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
        d_slot_moves(self.block_slot_keys, self.block_slot_keys_prime,
                     self.move_pattern)

        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments, using Thrust `transform`.
        d_extract_positions(self.block_slot_keys_prime, self.p_x_prime,
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
        evaled_block_count = d_evaluate_moves(self.omega_prime.row,
                                              self.omega_prime.col, self.p_x,
                                              self.p_x_prime, self._e_x,
                                              self._e_x2, self.p_y,
                                              self.p_y_prime, self._e_y,
                                              self._e_y2, self.r_inv, 1.59,
                                              self._block_keys,
                                              self.omega_prime.data)

        # Compute move-deltas using a Thrust `reduce_by_key` call over a
        # `transform` iterator.
        #d_minus_float(self.omega_prime.data, self._n_c, self.delta_n)
        delta_n = np.zeros(self.delta_n.size, dtype=self.delta_n.dtype)
        delta_n[self._block_keys[:evaled_block_count]] = (
            self.omega_prime.data[:evaled_block_count] -
            self._n_c[:evaled_block_count])
        self.delta_n[:] = delta_n
        #d_minus_float(self.omega_prime.data, self._n_c, self.delta_n)

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
        d_compute_block_group_keys(self.block_slot_keys,
                                   self.block_slot_keys_prime,
                                   self.block_group_keys,
                                   self.s2p.total_slot_count)

        # Thrust `reduce`.
        unmoved_count = d_equal_count_uint32(self.block_slot_keys,
                                             self.block_slot_keys_prime)

        # ## Packed block group keys ##
        #
        # Each block that has been assigned a non-zero move belongs to exactly
        # one group of associated-moves.  For each group of associated-moves,
        # there is a corresponding group key.  Given the index of a block in
        # the list of blocks belonging to groups of associated moves,

        # Thrust `sequence`.
        d_sequence_int32(self._group_block_keys)

        # Thrust `sort_by_key`. _(use `self._delta_s` as temporary, since it is
        # an array of the correct size)_.
        d_copy_int32(self.block_group_keys, self.block_group_keys_sorted)
        d_sort_netlist_keys(self.block_group_keys_sorted,
                            self._group_block_keys)

        if unmoved_count >= self._group_block_keys.size:
            return 0, self._group_block_keys.size

        # Thrust `inclusive_scan`.
        output_size = self._group_block_keys.size - unmoved_count
        d_permuted_nonmatch_inclusive_scan_int32(self.block_group_keys,
                                                 self._group_block_keys,
                                                 self._packed_block_group_keys,
                                                 output_size)

        # Thrust `reduce_by_key` over a `permutation` iterator.
        N = d_sum_permuted_float_by_key(self._packed_block_group_keys,
                                        self.delta_n, self._group_block_keys,
                                        self._block_keys, self._delta_s,
                                        output_size)

        # Thrust `copy_if`.
        N = d_assess_groups(temperature, self._group_block_keys,
                            self._packed_block_group_keys, self._delta_s,
                            self._rejected_block_keys, output_size)

        #      (moves evaluated)                , (moves rejected)
        return (self.block_slot_keys.size - unmoved_count), N


    @profile
    def apply_groups(self, rejected_move_block_count):
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
        if rejected_move_block_count == 0:
            return

        # Thrust `transform`.
        d_copy_permuted_uint32(self.block_slot_keys,
                               self.block_slot_keys_prime,
                               self._rejected_block_keys,
                               rejected_move_block_count)

        self.block_slot_keys, self.block_slot_keys_prime = (
            self.block_slot_keys_prime, self.block_slot_keys)

    @profile
    def run_iteration(self, seed, temperature, max_io_move=None,
                      max_logic_move=None):
        self.propose_moves(seed, max_io_move=max_io_move,
                           max_logic_move=max_logic_move)
        self.evaluate_moves()
        moved_count, rejected_moves = self.assess_groups(temperature)
        self.apply_groups(rejected_moves)
        self.evaluate_placement()
        return moved_count, rejected_moves

    def finalize(self):
        # Copy the final position of each block to the slot-to-block-key
        # mapping.
        self._sync_block_slot_keys_to_slot_block_keys()

        # To make output compatible with VPR, we must pack blocks in IO tiles
        # to fill IO tile-slots contiguously.
        io_slot_block_keys = self.slot_block_keys[:self.s2p.io_slot_count]
        pack_io(io_slot_block_keys, self.io_capacity)
        self.slot_block_keys[:self.s2p.io_slot_count] = \
            io_slot_block_keys[:]
        self._sync_slot_block_keys_to_block_slot_keys()

    def get_state(self):
        return {}
