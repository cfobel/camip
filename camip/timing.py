from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from cythrust.device_vector import (DeviceVectorInt8, DeviceVectorFloat32,
                                    DeviceVectorInt32)
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK,
                                                        LOGIC_BLOCK)
from cyplace_experiments.data import open_netlists_h5f
from .camip import (CAMIP, DeviceSparseMatrix)
from cyplace_experiments.data.connections_table import ConnectionsTable
from .device.CAMIP import (look_up_delay as d_look_up_delay,
                           sort_netlist_keys as d_sort_netlist_keys,
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
        sink_type = sink_connections.block_type
        driver_type = (driver_connections.block_type.as_matrix()
                       [sink_connections.net_key.as_matrix()])

        self.delay_type = DeviceVectorInt8.from_array((driver_type + 10 *
                                                       sink_type).as_matrix())

        # `arrival_matrix.row` holds driver block key.
        # `arrival_matrix.col` holds sink block key.
        # Initialize delays as 1 to compute unit delays.
        self.arrival_matrix = DeviceSparseMatrix(
            driver_connections.block_key.as_matrix()[sink_connections.net_key
                                                     .as_matrix()],
            sink_connections.block_key.as_matrix(),
            np.ones(len(sink_connections), dtype=np.float32))

        # `departure_matrix.row` holds sink block key.
        # `departure_matrix.col` holds driver block key.
        # Initialize delays as 1 to compute unit delays.
        self.departure_matrix = DeviceSparseMatrix(
            self.arrival_matrix.col[:],
            self.arrival_matrix.row[:],
            np.ones(self.arrival_matrix.data.size, dtype=np.float32))

        self.min_arrival_delays = DeviceVectorFloat32(len(sink_connections))
        self.max_arrival_delays = DeviceVectorFloat32(len(sink_connections))

        self.delta_x = DeviceVectorInt32(self.arrival_matrix.row.size)
        self.delta_y = DeviceVectorInt32(self.arrival_matrix.row.size)

        self.input_block_keys = self._connections_table.input_block_keys()
        self.output_block_keys = self._connections_table.output_block_keys()

        self.arrival_times = DeviceVectorFloat32(self.block_count)
        self.arrival_levels = DeviceVectorInt8(self.block_count)
        self.departure_times = DeviceVectorFloat32(self.block_count)
        self.departure_levels = DeviceVectorInt8(self.block_count)
        self.block_is_sync = DeviceVectorInt8(self.block_count)
        self.max_arrival_level = None

        # To compute arrival times, we need to perform a reduction connections
        # according to sink block.  Therefore, we need to sort the COO sparse
        # matrix indexes by sink block key _(i.e., column key)_.
        d_sort_netlist_keys(self.arrival_matrix.col, self.arrival_matrix.row)
        d_sort_netlist_keys(self.departure_matrix.col,
                            self.departure_matrix.row)
        self._compute_arrival_levels()

    def _compute_arrival_levels(self):
        arrival_times = np.zeros(self.block_count, dtype=np.float32)

        # Initialize the arrival time of input blocks to $\epsilon$ _(i.e.,
        # 1e-45)_.
        arrival_times[self.input_block_keys] = 1e-45

        # Initialize the arrival time of synchronous logic blocks to $\epsilon$
        # _(i.e., 1e-45)_.
        block_is_sink = np.zeros(self.block_count, dtype=bool)
        block_is_sink[self.input_block_keys] = True
        block_is_sink[self.output_block_keys] = True
        connections = self._connections_table.connections
        block_is_sink[connections[(connections.block_type == LOGIC_BLOCK) &
                                  (connections.type ==
                                   CONNECTION_SINK)].block_key.unique()] = True
        arrival_times[~block_is_sink] = 1e-45

        self.arrival_times[:] = arrival_times
        del arrival_times

        block_is_sync = np.zeros(self.block_count, dtype=np.int8)
        block_is_sync[self._connections_table.synchronous_blocks] = 1
        self.block_is_sync[:] = block_is_sync
        del block_is_sync
        self.max_arrival_level = self.update_arrival_times()
        self.arrival_levels[:] = self.arrival_times[:]

    @profile
    def run_iteration(self, seed, temperature, max_io_move=None,
                      max_logic_move=None):
        # TODO: Determine the number of times to loop based on the maximum unit
        # arrival time delay.
        self.update_connection_delays(self.arrival_matrix, self.delays)
        self.update_arrival_times()
        return super(CAMIPTiming, self).run_iteration(seed, temperature,
                                                      max_io_move,
                                                      max_logic_move)

    @profile
    def update_connection_delays(self, matrix):
        d_look_up_delay(matrix.row, matrix.col,
                        self.p_x, self.p_y, self.delays, self.nrows,
                        self.ncols, self.delay_type,
                        self.delta_x, self.delta_y,
                        matrix.data)

    @profile
    def update_arrival_times(self):
        arrival_times = self.arrival_times[:]
        arrival_times[arrival_times > 1e-45] = 0
        self.arrival_times[:] = arrival_times

        i = 0

        # Before the maximum arrival "level" _(i.e., max unit-delay arrival
        # time)_ has been determined, we need to continue until an arrival time
        # has been resolved for each block _(i.e., `arrival_times.min() > 0`)_.
        while ((self.max_arrival_level is None and (arrival_times.min() <= 0))
               or i < self.max_arrival_level):
            N = d_arrival_delays(self.arrival_matrix.col,
                                 self.arrival_matrix.row, self.block_is_sync,
                                 self.arrival_times, self.arrival_matrix.data,
                                 self._block_keys, self.min_arrival_delays,
                                 self.max_arrival_delays)
            arrival_times = self.arrival_times[:]
            ready_sinks = (self.min_arrival_delays[:N] > 0)
            arrival_times[self._block_keys[:][ready_sinks]] = \
                self.max_arrival_delays[:][ready_sinks]
            #print N, (arrival_times > 0).sum()
            self.arrival_times[:] = arrival_times
            i += 1
            if i > 30:
                raise RuntimeError('Failed to compute arrival times after %d '
                                   'iterations.' % i)
        return i

    def plot_arrival_levels(self):
        fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
        self.plot_integer(axes, self.arrival_levels[:])
        axes[0].set_title('arrival levels (unit-delay arrival times)')
        return fig

    def plot_arrival_times(self):
        fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
        self.plot_float(axes, self.arrival_times[:])
        axes[0].set_title('arrival times')
        return fig

    def plot_arrival_delays_ij(self):
        fig, axes = self.fig_from_gridspec(GridSpec(2, 1), 2)
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
