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
from cyplace_experiments.data import open_netlists_h5f
from cyplace_experiments.data.connections_table import (INPUT_BLOCK,
                                                        OUTPUT_BLOCK,
                                                        LOGIC_BLOCK,
                                                        CONNECTION_CLOCK,
                                                        CONNECTION_CLOCK_DRIVER)
from .CAMIP import (VPRAutoSlotKeyTo2dPosition, random_vpr_pattern,
                    cAnnealSchedule, get_std_dev, pack_io,
                    # Only used in class constructors _(not in main methods)_.
                    sum_float_by_key)

from cythrust.device_vector import (DeviceVectorInt32, DeviceVectorFloat32,
                                    DeviceVectorUint32)
from cythrust import DeviceDataFrame, DeviceVectorCollection
import cythrust.device_vector as dv
import device.CAMIP as _CAMIP
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
                          permuted_nonmatch_inclusive_scan_int32 as
                          d_permuted_nonmatch_inclusive_scan_int32,
                          assess_groups as d_assess_groups,
                          sum_float_by_key as d_sum_float_by_key,
                          copy_int32 as d_copy_int32,
                          permuted_fill_float32 as d_permuted_fill_float32,
                          permuted_fill_int32 as d_permuted_fill_int32,
                          look_up_delay as d_look_up_delay,
                          )


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
            deltas = np.abs(placer.block_link_data['cost_prime'][:block_count]
                            - placer.block_link_data['cost'][:block_count])
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
        total_move_count = 0
        #while (self.anneal_schedule.temperature > 0.00001 * placer.theta /
               #placer.net_count):
        while (not placer.exit_criteria(self.anneal_schedule.temperature)):
            start = time.time()
            total_moves, rejected_moves = self.outer_iteration(placer)
            end = time.time()
            total_move_count += total_moves
            extra_state = placer.get_state()
            state = OrderedDict([('start', start), ('end', end),
                                 ('cost', placer.theta),
                                 ('temperature',
                                  self.anneal_schedule.temperature),
                                 ('success_ratio',
                                  self.anneal_schedule.success_ratio),
                                 ('radius_limit', self.anneal_schedule.rlim),
                                 ('total_iteration_count', total_move_count)])
            state.update(extra_state)
            if not states:
                print '\n| ' + ' | '.join(state.keys()[2:]) + ' |'
                print '|' + '|'.join(['-' * (len(k) + 2)
                                      for k in state.keys()[2:]]) + '|'
            states.append(state)
            print '|' + '|'.join(['{0:>{1}s}'.format(si_format(v, 2),
                                                     len(k) + 2)
                                  for k, v in state.items()[2:]]) + '|'
        print '\nRuntime: %.2fs' % (states[-1]['end'] - states[0]['start'])
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


class Placement(object):
    '''
    This class holds a representation of the placement in two data structures.

    The first data structure contains per-block data, including:

     - The block label.
     - The block type.
     - The block slot assignment.

    The second data structure contains per-slot data, including:

     - The slot type.
     - The slot assignment.
    '''

    def __init__(self, block_data, io_capacity, allocator=dv):
        '''
        The block data frame must be of the following form:

        | Block label | Block type   |
        |-------------|--------------|
        | b_out0      | OUTPUT_BLOCK |
        | b_out1      | OUTPUT_BLOCK |
        | b_in0       | INPUT_BLOCK  |
        | b_in1       | INPUT_BLOCK  |

        Internally, this is converted to a purely numeric form in a
        `DeviceDataFrame`.
        '''
        self._block_data = (block_data[['block_key', 'block_type',
                                        'block_label']].sort('block_key')
                            .drop_duplicates('block_key')
                            .set_index('block_key'))
        del block_data
        self.block_data = DeviceDataFrame(self._block_data[['block_type']],
                                          allocator=allocator)
        self.block_data.add('block_key',
                            self._block_data.index.values.astype(np.int32))
        self.block_data.add('slot_key', dtype=np.uint32)

        type_counts = self._block_data.groupby('block_type').apply(lambda x:
                                                                   len(x))
        self.input_count = type_counts.loc[INPUT_BLOCK]
        self.output_count = type_counts.loc[OUTPUT_BLOCK]
        self.s2p = VPRAutoSlotKeyTo2dPosition(self.input_count +
                                              self.output_count,
                                              type_counts[LOGIC_BLOCK],
                                              io_capacity)
        self.slot_data = DeviceDataFrame({'block_key':
                                          np.empty(self.s2p.total_slot_count,
                                                   dtype=np.uint32)},
                                         allocator=allocator)
        self.slot_data.v['block_key'][:] = self.block_count

        self.block_type_keys = OrderedDict([
            (k, allocator.from_array(self._block_data
                                     .loc[self._block_data.block_type ==
                                          eval('%s_BLOCK' % k.upper())].index
                                     .values.astype(np.int32)))
            for k in ('input', 'output', 'logic')])

        slot_block_keys = self.slot_data['block_key'][:]
        # Fill IO slots.
        slot_block_keys[:self.input_count] = self.block_type_keys['input'][:]
        slot_block_keys[self.input_count:self.io_count] = \
            self.block_type_keys['output'][:]

        ## Fill logic slots.
        slot_block_keys[self.io_slot_count:self.io_slot_count +
                        self.logic_count] = self.block_type_keys['logic'][:]
        self.slot_data.v['block_key'][:] = slot_block_keys
        self._sync_slot_block_keys_to_block_slot_keys()

    def block_positions(self, allocator=dv):
        # Pack IO in each tile to lowest Z-indices.
        self._pack_io()
        position_data = DeviceDataFrame({'p_x': np.zeros(self.block_count,
                                                         dtype=np.int32)},
                                        allocator=allocator)
        position_data.add('p_y', dtype=np.int32)
        position_data.add('p_z', np.zeros(self.block_count, dtype=np.int32))
        b = position_data
        _CAMIP.extract_positions(self.block_data.v['slot_key'], b.v['p_x'],
                                 b.v['p_y'], self.s2p)
        b.v['p_z'][:self.io_count] = (self.block_data['slot_key']
                                      [:self.io_count] % self.io_capacity)
        return position_data

    @property
    def io_capacity(self):
        return self.s2p.io_capacity

    @property
    def io_slot_count(self):
        return self.s2p.io_slot_count

    @property
    def block_count(self):
        return self.io_count + self.logic_count

    @property
    def io_count(self):
        return self.s2p.io_count

    @property
    def logic_count(self):
        return self.s2p.logic_count

    def shuffle(self):
        '''
        Shuffle placement permutation.

        The shuffle is aware of IO and logic slots in the placement, and will
        keep IO and logic within the corresponding areas of the permutation.
        '''
        slot_block_keys = self.slot_data['block_key'][:]
        np.random.shuffle(slot_block_keys[:self.s2p.io_slot_count].values)
        np.random.shuffle(slot_block_keys[self.s2p.io_slot_count:].values)
        self.slot_data.v['block_key'][:] = slot_block_keys[:]
        self._sync_slot_block_keys_to_block_slot_keys()

    def _sync_block_slot_keys_to_slot_block_keys(self):
        '''
        Update `slot_block_keys` based on `block_slot_keys`.

        Useful when updating the permutation slot contents directly _(e.g.,
        shuffling the contents of the permutation slots)_.
        '''
        slot_block_keys = self.slot_data['block_key'][:]
        slot_block_keys[:] = self.block_count
        slot_block_keys[self.block_data['slot_key'].values] = \
            self.block_data['block_key'].values
        self.slot_data.v['block_key'][:] = slot_block_keys[:].values

    def _sync_slot_block_keys_to_block_slot_keys(self):
        '''
        Update `block_slot_keys` based on `slot_block_keys`.
        '''
        occupied = np.where(self.slot_data['block_key'].values < self.block_count)
        block_slot_keys = self.block_data['slot_key'].values
        slot_block_keys = self.slot_data['block_key'].values
        block_slot_keys[slot_block_keys[occupied[0]]] = occupied[0]
        self.block_data.v['slot_key'][:] = block_slot_keys

    @property
    def slot_block_keys(self):
        self._pack_io()
        return self.slot_data.v['block_key']

    def _pack_io(self):
        # Copy the final position of each block to the slot-to-block-key
        # mapping.
        self._sync_block_slot_keys_to_slot_block_keys()

        # To make output compatible with VPR, we must pack blocks in IO tiles
        # to fill IO tile-slots contiguously.
        slot_block_keys = self.slot_data['block_key'][:]
        pack_io(slot_block_keys[:self.io_slot_count].values,
                self.s2p.io_capacity)
        self.slot_data.v['block_key'][:] = slot_block_keys[:]
        self._sync_slot_block_keys_to_block_slot_keys()


def partition_keys(connections, mask=None, drop_exclude=False):
    # # `partition_keys` #
    #
    # Partition keys and reassign net and block keys such that the ranges keys
    # for the first partition are contiguous.
    #
    #        0                                 partition size
    #        |                                        |
    #        ╔════════════════════════════════════════╗┌──────────────────┐
    #        ║ `mask=True`                            ║│ `mask=False`     │
    #        ╚════════════════════════════════════════╝└──────────────────┘
    if mask is not None:
        connections['exclude'] = ~mask
    elif 'exclude' not in connections:
        raise KeyError('If `mask` is not specified, `exclude` must exist.')
    key_connections = connections.sort(['exclude', 'net_key'])
    key_connections.drop_duplicates('net_key', inplace=True)
    key_connections['new_net_key'] = np.arange(
        key_connections['net_key'].shape[0],
        dtype=key_connections['net_key'].dtype)
    key_connections.set_index('net_key', inplace=True)
    connections.loc[:, 'net_key'] = (key_connections.loc
                                     [connections['net_key'].values]
                                     ['new_net_key'].values.ravel())
    key_connections = connections.sort(['exclude', 'block_key'])
    key_connections.drop_duplicates('block_key', inplace=True)
    key_connections['new_block_key'] = np.arange(
        key_connections['block_key'].shape[0],
        dtype=key_connections['block_key'].dtype)
    key_connections.set_index('block_key', inplace=True)
    connections.loc[:, 'block_key'] = (key_connections.loc
                                       [connections['block_key'].values]
                                       ['new_block_key'].values.ravel())
    if drop_exclude:
        return connections.loc[~connections['exclude']]
    else:
        return connections.sort('exclude')


class CAMIP(object):
    def __init__(self, connections, placement, allocator=dv):
        '''
        Arguments:

          - `connections`:
           * A `pandas.DataFrame` object, with the following columns:

                | net key | block key |
                |---------|-----------|
                |    n    |     b     |

           * __NB__ The net keys and block keys must be contiguous ranges
             starting at zero.  This is necessary because we do
             scattering/gathering based on unique keys in the table.
           * Use `partition_keys` if you'd like to exclude some connections,
             e.g., clock connections, that might break the contiguous key
             ranges.
          - `placement`: `Placement` instance.
        '''
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
        #if include_clock:
            #mask = np.ones(connections_table.connections.shape[0], dtype=np.bool)
        #else:
            #mask = ~connections_table.connections.type.isin([CONNECTION_CLOCK,
                                                             #CONNECTION_CLOCK_DRIVER])
        block_keys = np.sort(connections.block_key.unique())
        self.s2p = placement.s2p
        self.CAMIP = _CAMIP
        self.block_count = block_keys.size
        self.net_count = connections.net_key.unique().size

        # Declare device vector frame only for block keys that are listed in
        # the provided `connections`.
        # __NB__ This may be a subset of the blocks in the placement.
        self.block_data = DeviceDataFrame({'block_key':
                                           block_keys.astype(np.uint32),
                                           'slot_key':
                                           placement.block_data['slot_key'][:]
                                           .values[block_keys]
                                           .astype(np.uint32)},
                                          allocator=allocator)
        self.block_data.add('slot_key_prime', dtype=np.uint32)
        self.block_data.add('p_x', dtype=np.int32)
        self.block_data.add('p_y', dtype=np.int32)
        self.block_data.add('p_x_prime', dtype=np.int32)
        self.block_data.add('p_y_prime', dtype=np.int32)
        self.block_data.add('delta_cost', dtype=np.float32)
        self.block_data.add('group_key', dtype=np.int32)
        self.block_data.add('sorted_group_key', dtype=np.int32)
        self.block_data.add('packed_group_key', dtype=np.int32)
        self.block_data.add('rejected_block_key', dtype=np.int32)

        self.group_data = DeviceDataFrame({'block_key':
                                           np.empty(self.block_count,
                                                    dtype=np.int32)},
                                          allocator=allocator)
        self.group_data.add('delta_cost', dtype=np.float32)

        self.net_link_data = DeviceDataFrame(
            OrderedDict([('net_key', connections.net_key.values
                          .astype(np.int32)),
                         ('block_key', connections.block_key.values
                          .astype(np.int32))]), allocator=allocator)
        self.CAMIP.sort_int32_by_int32_key(self.net_link_data.v['net_key'],
                                           self.net_link_data.v['block_key'])

        self.net_link_data.add('x', dtype=np.float32)
        self.net_link_data.add('x2', dtype=np.float32)
        self.net_link_data.add('y', dtype=np.float32)
        self.net_link_data.add('y2', dtype=np.float32)
        self.net_link_data.add('cost', dtype=np.float32)
        self.net_link_data.add('reduced_keys', dtype=np.int32)

        self.block_link_data = DeviceDataFrame(
            OrderedDict([('block_key', connections.block_key.values
                          .astype(np.int32)),
                         ('net_key', connections.net_key.values
                          .astype(np.int32))]), allocator=allocator)

        self.CAMIP.sort_int32_by_int32_key(self.block_link_data.v['block_key'],
                                           self.block_link_data.v['net_key'])
        self.block_link_data.add('cost', dtype=np.float32)
        self.block_link_data.add('cost_prime', dtype=np.float32)

        # Declare device frame only for net keys that are listed in the
        # provided `connections`.
        # __NB__ This may be a subset of the nets in the placement.
        self.net_data = DeviceDataFrame({'r_inv': np.empty(self.net_count,
                                                           dtype=np.float32)},
                                        allocator=allocator)
        self.net_data.add('block_count', dtype=np.int32)

        # Add temporary columns to net connections/links data-frame table to
        # compute the number of blocks connected to each net.
        self.net_link_data.add('ones', np.ones(self.net_link_data.size,
                                               dtype=np.int32))
        self.net_link_data.add('reduced_block_count', dtype=np.int32)

        self.CAMIP.sum_int32_by_int32_key(self.net_link_data.v['net_key'],
                                          self.net_link_data.v['ones'],
                                          self.net_link_data.v['reduced_keys'],
                                          self.net_link_data.v['reduced_block_count'])

        self.net_data.v['block_count'][:] = (self.net_link_data
                                             ['reduced_block_count']
                                             [:self.net_count])
        self.net_data.v['r_inv'][:] = np.reciprocal(self.net_link_data
                                                    ['reduced_block_count']
                                                    [:self.net_count]
                                                    .astype(np.float32))
        # Drop the temporary columns from the data frame, since we don't need
        # them any more.
        self.net_link_data.drop('reduced_block_count')
        self.net_link_data.drop('ones')

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
        b = self.block_data
        nc = self.net_link_data
        n = self.net_data
        bc = self.block_link_data

        self.CAMIP.extract_positions(b.v['slot_key'], b.v['p_x'], b.v['p_y'],
                                     self.s2p)

        # Star+ vectors
        # Thrust `reduce_by_key`.
        self.CAMIP.sum_xy_vectors(nc.v['block_key'], nc.v['net_key'],
                                  b.v['p_x'], b.v['p_y'], nc.v['x'],
                                  nc.v['x2'], nc.v['y'], nc.v['y2'],
                                  nc.v['reduced_keys'])

        # `theta`: $\theta =$ total placement cost
        # Thrust `transform`.
        self.theta = self.CAMIP.star_plus_2d(nc.v['x'], nc.v['x2'], nc.v['y'],
                                             nc.v['y2'], n.v['r_inv'], 1.59,
                                             nc.v['cost'])

        # $\vec{n_c}$ contains the total cost of all edges connected to node
        # $i$.
        # Thrust `reduce_by_key`.
        self.CAMIP.sum_permuted_float_by_key(bc.v['block_key'], nc.v['cost'],
                                             bc.v['net_key'],
                                             nc.v['reduced_keys'],
                                             bc.v['cost'], bc.size)
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
        self.CAMIP.slot_moves(self.block_data.v['slot_key'],
                              self.block_data.v['slot_key_prime'],
                              self.move_pattern)

        # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on
        # permutation slot assignments, using Thrust `transform`.
        self.CAMIP.extract_positions(self.block_data.v['slot_key_prime'],
                                     self.block_data.v['p_x_prime'],
                                     self.block_data.v['p_y_prime'], self.s2p)

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
        bc = self.block_link_data
        b = self.block_data
        nc = self.net_link_data
        n = self.net_data
        self.CAMIP.evaluate_moves(bc.v['block_key'], bc.v['net_key'],
                                  b.v['p_x'], b.v['p_x_prime'], nc.v['x'],
                                  nc.v['x2'], b.v['p_y'], b.v['p_y_prime'],
                                  nc.v['y'], nc.v['y2'], n.v['r_inv'], 1.59,
                                  nc.v['reduced_keys'], bc.v['cost_prime'])

        # Compute move-deltas using a Thrust `reduce_by_key` call over a
        # `transform` iterator.
        self.CAMIP.minus_float(bc.v['cost_prime'], bc.v['cost'],
                               b.v['delta_cost'])

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
        b = self.block_data
        g = self.group_data

        # Thrust `transform`.
        self.CAMIP.compute_block_group_keys(b.v['slot_key'],
                                            b.v['slot_key_prime'],
                                            b.v['group_key'],
                                            self.s2p.total_slot_count)

        # Thrust `reduce`.
        unmoved_count = self.CAMIP.equal_count_uint32(b.v['slot_key'],
                                                      b.v['slot_key_prime'])

        # ## Packed block group keys ##
        #
        # Each block that has been assigned a non-zero move belongs to exactly
        # one group of associated-moves.  For each group of associated-moves,
        # there is a corresponding group key.  Given the index of a block in
        # the list of blocks belonging to groups of associated moves,

        # Thrust `sequence`.
        self.CAMIP.sequence_int32(g.v['block_key'])

        # Thrust `sort_by_key`. _(use `self._delta_s` as temporary, since it is
        # an array of the correct size)_.
        self.CAMIP.copy_int32(b.v['group_key'], b.v['sorted_group_key'])
        self.CAMIP.sort_int32_by_int32_key(b.v['sorted_group_key'],
                                           g.v['block_key'])

        if unmoved_count >= g.size:
            return 0, g.size

        # Thrust `inclusive_scan`.
        output_size = g.size - unmoved_count
        self.CAMIP.permuted_nonmatch_inclusive_scan_int32(
            b.v['group_key'], g.v['block_key'], b.v['packed_group_key'],
            output_size)

        # Thrust `reduce_by_key` over a `permutation` iterator.
        N = self.CAMIP.sum_permuted_float_by_key(
            b.v['packed_group_key'], b.v['delta_cost'], g.v['block_key'],
            self.net_link_data.v['reduced_keys'], g.v['delta_cost'],
            output_size)

        # Thrust `copy_if`.
        N = self.CAMIP.assess_groups(temperature, g.v['block_key'],
                                     b.v['packed_group_key'],
                                     g.v['delta_cost'],
                                     b.v['rejected_block_key'], output_size)

        #      (moves evaluated)                , (moves rejected)
        return (b.size - unmoved_count), N

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

        b = self.block_data
        # Thrust `transform`.
        self.CAMIP.copy_permuted_uint32(b.v['slot_key'], b.v['slot_key_prime'],
                                        b.v['rejected_block_key'],
                                        rejected_move_block_count)

        b.v['slot_key'], b.v['slot_key_prime'] = (b.v['slot_key_prime'],
                                                  b.v['slot_key'])

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

    def get_state(self):
        return {}
