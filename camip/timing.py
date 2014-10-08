from collections import Counter
import time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from cythrust.device_vector import (DeviceVectorInt8, DeviceVectorFloat32,
                                    DeviceVectorInt32)
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK,
                                                        LOGIC_BLOCK)
from cyplace_experiments.data import open_netlists_h5f
from .camip import (CAMIP, DeviceSparseMatrix)
from cyplace_experiments.data.connections_table import ConnectionsTable
from .device.CAMIP import (look_up_delay as d_look_up_delay,
                           sort_delay_matrix as d_sort_delay_matrix,
                           arrival_delays as d_arrival_delays)

try:
    profile
except:
    profile = lambda (f): f


class CAMIPTiming(CAMIP):
    def __init__(self, net_list_name, io_capacity=3):
        super(CAMIPTiming, self).__init__(ConnectionsTable(net_list_name),
                                          io_capacity)
        h5f = open_netlists_h5f()

        self.nrows, self.ncols = self.s2p.extent
        arch = getattr(h5f.root.architectures.vpr__k4_n1, 'x%04d_by_y%04d' %
                       (self.nrows, self.ncols))

        self.delays = DeviceVectorFloat32(np.sum([np.prod(c.shape)
                                                  for c in arch]))

        offset = 0

        for name in ('fb_to_fb', 'fb_to_io', 'io_to_fb', 'io_to_io'):
            c = getattr(arch, name)
            c_size = np.prod(c.shape)
            self.delays[offset:offset + c_size] = c[:].ravel()
            offset += c_size
        del arch
        h5f.close()

        connections = self._connections_table.connections
        driver_connections = connections[connections.type ==
                                         CONNECTION_DRIVER].sort('net_key')
        sink_connections = connections[connections.type == CONNECTION_SINK]

        # `arrival_matrix.row` holds driver block key.
        # `arrival_matrix.col` holds sink block key.
        # Initialize delays as 1 to compute unit delays.
        self.arrival_matrix = DeviceSparseMatrix(
            driver_connections.block_key.as_matrix()[sink_connections.net_key
                                                     .as_matrix()],
            sink_connections.block_key.as_matrix(),
            np.ones(len(sink_connections), dtype=np.float32))

        sink_type = sink_connections.block_type
        driver_type = (driver_connections.block_type.as_matrix()
                       [sink_connections.net_key.as_matrix()])
        self.arrival_delay_type = DeviceVectorInt8.from_array(
            (driver_type + 10 * sink_type).as_matrix())
        self.departure_delay_type = DeviceVectorInt8.from_array(
            self.arrival_delay_type[:])

        # `departure_matrix.row` holds sink block key.
        # `departure_matrix.col` holds driver block key.
        # Initialize delays as 1 to compute unit delays.
        self.departure_matrix = DeviceSparseMatrix(
            self.arrival_matrix.col[:],
            self.arrival_matrix.row[:],
            np.ones(self.arrival_matrix.data.size, dtype=np.float32))

        self.min_delays = DeviceVectorFloat32(len(sink_connections))
        self.max_delays = DeviceVectorFloat32(len(sink_connections))

        self.delta_x = DeviceVectorInt32(self.arrival_matrix.row.size)
        self.delta_y = DeviceVectorInt32(self.arrival_matrix.row.size)

        self.input_block_keys = self._connections_table.input_block_keys()
        self.output_block_keys = self._connections_table.output_block_keys()

        self.arrival_times = DeviceVectorFloat32(self.block_count)
        self.arrival_levels = DeviceVectorInt8(self.block_count)
        self.departure_times = DeviceVectorFloat32(self.block_count)
        self.departure_levels = DeviceVectorInt8(self.block_count)
        self.max_delay_level = None

        self.block_is_sync = DeviceVectorInt8(self.block_count)
        block_is_sync = np.zeros(self.block_count, dtype=np.int8)
        block_is_sync[self._connections_table.synchronous_blocks] = 1
        self.block_is_sync[:] = block_is_sync
        del block_is_sync

        # To compute arrival times, we need to perform a reduction connections
        # according to sink block.  Therefore, we need to sort the COO sparse
        # matrix indexes by sink block key _(i.e., column key)_.
        d_sort_delay_matrix(self.arrival_matrix.col, self.arrival_matrix.row,
                            self.arrival_delay_type)
        d_sort_delay_matrix(self.departure_matrix.col,
                            self.departure_matrix.row,
                            self.departure_delay_type)
        self.arrival_matrix.delay_type = self.arrival_delay_type
        self.departure_matrix.delay_type = self.departure_delay_type
        self.max_delay_level = self.compute_delay_levels(self.arrival_times,
                                                         self.arrival_matrix)
        self.arrival_levels[:] = self.arrival_times[:]

    def compute_delay_levels(self, longest_delays_v, delays_matrix):
        longest_delays = np.zeros(self.block_count, dtype=np.float32)

        # Initialize the arrival time of input blocks to $\epsilon$ _(i.e.,
        # 1e-45)_.
        longest_delays[self.input_block_keys] = 1e-45
        longest_delays[self.output_block_keys] = 1e-45

        # Initialize the arrival time of synchronous logic blocks to $\epsilon$
        # _(i.e., 1e-45)_.
        block_is_sink = np.zeros(self.block_count, dtype=bool)
        block_is_sink[self.input_block_keys] = True
        block_is_sink[self.output_block_keys] = True
        connections = self._connections_table.connections
        block_is_sink[connections[(connections.block_type == LOGIC_BLOCK) &
                                  (connections.type ==
                                   CONNECTION_SINK)].block_key.unique()] = True
        longest_delays[~block_is_sink] = 1e-45

        longest_delays_v[:] = longest_delays
        del longest_delays

        return self.update_longest_delay_times_from_scratch(longest_delays_v,
                                                            delays_matrix)

    @profile
    def run_iteration(self, seed, temperature, max_io_move=None,
                      max_logic_move=None):
        # TODO: Determine the number of times to loop based on the maximum unit
        # arrival time delay.
        self.update_connection_delays(self.arrival_matrix)
        self.update_longest_delay_times_from_scratch(self.arrival_times,
                                                     self.arrival_matrix)
        return super(CAMIPTiming, self).run_iteration(seed, temperature,
                                                      max_io_move,
                                                      max_logic_move)

    @profile
    def update_connection_delays(self, matrix):
        d_look_up_delay(matrix.row, matrix.col,
                        self.p_x, self.p_y, self.delays, self.nrows,
                        self.ncols, matrix.delay_type,
                        self.delta_x, self.delta_y,
                        matrix.data)

    @profile
    def update_longest_delay_times_from_scratch(self, longest_delays_v,
                                                delays_matrix):
        longest_delays = longest_delays_v[:]
        longest_delays[longest_delays > 1e-45] = 0
        longest_delays_v[:] = longest_delays

        i = 0

        # Before the maximum arrival "level" _(i.e., max unit-delay arrival
        # time)_ has been determined, we need to continue until an arrival time
        # has been resolved for each block _(i.e., `longest_delays_v.min() >
        # 0`)_.
        while longest_delays.min() <= 0:
            N = d_arrival_delays(delays_matrix.col,
                                 delays_matrix.row, self.block_is_sync,
                                 longest_delays_v, delays_matrix.data,
                                 self._block_keys, self.min_delays,
                                 self.max_delays)
            longest_delays = longest_delays_v[:]
            ready_sinks = (self.min_delays[:N] > 0)
            longest_delays[self._block_keys[:][ready_sinks]] = \
                self.max_delays[:][ready_sinks]
            longest_delays_v[:] = longest_delays
            i += 1
            if i > 30:
                import pudb; pudb.set_trace()
                raise RuntimeError('Failed to compute arrival times after %d '
                                   'iterations.' % i)
        return i

    def plot_arrival_levels(self, fig=None):
        if fig is None:
            fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
        else:
            axes = fig.axes
        self.plot_integer(axes, self.arrival_levels[:])
        axes[0].set_title('arrival levels (unit-delay arrival times)')
        return fig

    def plot_arrival_times(self, fig=None):
        if fig is None:
            fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
        else:
            axes = fig.axes
        self.plot_float(axes, self.arrival_times[:])
        axes[0].set_title('arrival times')
        return fig

    def plot_arrival_delays_ij(self, fig=None):
        if fig is None:
            fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
        else:
            axes = fig.axes
        self.plot_float(axes, self.arrival_matrix.data[:])
        axes[0].set_title('arrival delays(i, j)')
        return fig

    def plot_integer(self, axes, values):
        axes[0].plot(values)
        axes[1].bar(*zip(*Counter(values).items()))

    def plot_float(self, axes, values):
        axes[0].plot(values)
        axes[1].hist(values)

    def fig_from_gridspec(self, gridspec, count, *args, **kwargs):
        fig = plt.figure(*args, **kwargs)
        axes = [fig.add_subplot(gridspec[i]) for i in range(count)]
        return fig, axes

    def draw(self):
        if not hasattr(self, 'fig'):
            self.count = 0
            self.gridspec = GridSpec(2, 1)
            self.fig = plt.figure()
            values = self.arrival_times[:]
            self.arrival_times_lim = values.min(), values.max()
            values = self._e_c[:self.net_count]
            self.block_cost_lim = values.min(), values.max()

        self.count += 1
        if self.count % 50 == 0:
            self.fig.clf()
            values = self.arrival_times[:]
            top = self.fig.add_subplot(self.gridspec[0],
                                       ylim=self.arrival_times_lim)
            colors = top._get_lines.color_cycle
            top.plot(values, linewidth=1, color=colors.next())
            self.fig.add_subplot(self.gridspec[1],
                                 xlim=self.arrival_times_lim).hist(values)
            block_cost = top.twinx()
            block_cost.set_ylim(self.block_cost_lim)
            block_cost.plot(self._e_c[:self.net_count], linewidth=1,
                            color=colors.next())
            plt.draw()
