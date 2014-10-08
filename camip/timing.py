import numpy as np
from cythrust.device_vector import (DeviceVectorInt8, DeviceVectorFloat32)
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
from cyplace_experiments.data import open_netlists_h5f
from .camip import (CAMIP, DeviceSparseMatrix)
from cyplace_experiments.data.connections_table import ConnectionsTable
from .device.CAMIP import look_up_delay as d_look_up_delay


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

        # `arrival_matrix.row` holds driver block key.
        # `arrival_matrix.col` holds sink block key.
        self.arrival_matrix = DeviceSparseMatrix(
            driver_connections.block_key.as_matrix()[sink_connections.net_key
                                                     .as_matrix()],
            sink_connections.block_key.as_matrix(),
            np.empty(len(sink_connections), dtype=np.float32))
        self.delay_type = DeviceVectorInt8.from_array((driver_type + 10 *
                                                       sink_type).as_matrix())
        # Initialize delays as 1 to compute unit delays.
        self.delays_ij = DeviceVectorFloat32.from_array(
            np.ones(self.arrival_matrix.row.size, dtype=np.float32))

    def update_connection_delays(self):
        d_look_up_delay(self.arrival_matrix.row, self.arrival_matrix.col,
                        self.p_x, self.p_y, self.delays, self.nrows,
                        self.ncols, self.delay_type, self.delays_ij)
