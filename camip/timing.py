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
                           arrival_delays as d_arrival_delays,
                           sort_delay_matrix_by_levels as
                           d_sort_delay_matrix_by_levels,
                           inclusive_scan_int32 as d_inclusive_scan_int32,
                           count_int8_key as d_count_int8_key,
                           )
from thrust_timing.path_timing import PathTimingData, get_arch_data
from cyplace_experiments.data.connections_table import (CONNECTION_DRIVER,
                                                        CONNECTION_SINK)
import cythrust.device_vector as dv


try:
    profile
except:
    profile = lambda (f): f


class CAMIPTiming(CAMIP):
    def __init__(self, connections_table, io_capacity=3):
        super(CAMIPTiming, self).__init__(connections_table, io_capacity)

        self.arrival_data = PathTimingData(get_arch_data(*self.s2p.extent),
                                           connections_table,
                                           source=CONNECTION_DRIVER)
        self.departure_data = PathTimingData(get_arch_data(*self.s2p.extent),
                                             connections_table,
                                             source=CONNECTION_SINK)

    def arrival_times(self):
        result = self.arrival_data.update_position_based_longest_paths(
            {'p_x': dv.view_from_vector(self.p_x),
             'p_y': dv.view_from_vector(self.p_y)})
        return result

    def departure_times(self):
        result = self.departure_data.update_position_based_longest_paths(
            {'p_x': dv.view_from_vector(self.p_x),
             'p_y': dv.view_from_vector(self.p_y)})
        return result

    @profile
    def run_iteration(self, seed, temperature, max_io_move=None,
                      max_logic_move=None):
        arrival_times = self.arrival_times()
        departure_times = self.departure_times()
        return super(CAMIPTiming, self).run_iteration(seed, temperature,
                                                      max_io_move,
                                                      max_logic_move)

