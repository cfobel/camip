import numpy as np
from fpga_netlist.connections_table import CONNECTION_DRIVER, CONNECTION_SINK
from thrust_timing.path_timing import PathTimingData, get_arch_data
import cythrust.device_vector as dv
from thrust_timing.SORT_TIMING import (look_up_delay_prime,
                                       block_delta_timing_cost,
                                       connection_cost,
                                       compute_normalized_weighted_sum)
from cythrust import DeviceDataFrame
from .camip import CAMIP


try:
    profile
except:
    profile = lambda (f): f


class CAMIPTiming(CAMIP):
    def __init__(self, connections_table, io_capacity=3, include_clock=False,
                 timing_cost_disabled=False, wire_length_factor=0.5,
                 criticality_exp=10.):
        self.timing_cost_disabled = timing_cost_disabled
        self.wire_length_factor = wire_length_factor
        super(CAMIPTiming, self).__init__(connections_table, io_capacity,
                                          include_clock=include_clock)

        self.arrival_data = PathTimingData(get_arch_data(*self.s2p.extent),
                                           connections_table,
                                           source=CONNECTION_DRIVER)
        self.departure_data = PathTimingData(get_arch_data(*self.s2p.extent),
                                             connections_table,
                                             source=CONNECTION_SINK)
        self.arrival_data.connections.add('cost', dtype=np.float32)
        self.arrival_data.connections.add('cost_prime', dtype=np.float32)
        self.arrival_data.connections.add('reduced_target_cost',
                                          dtype=np.float32)
        self.arrival_data.connections.add('delay_prime', dtype=np.float32)
        self.departure_data.connections.add('cost', dtype=np.float32)
        self.departure_data.connections.add('cost_prime', dtype=np.float32)
        self.departure_data.connections.add('reduced_target_cost',
                                            dtype=np.float32)
        self.departure_data.connections.add('delay_prime', dtype=np.float32)
        self.position_views = {'p_x': dv.view_from_vector(self.p_x),
                               'p_y': dv.view_from_vector(self.p_y),
                               'p_x_prime':
                               dv.view_from_vector(self.p_x_prime),
                               'p_y_prime':
                               dv.view_from_vector(self.p_y_prime)}

        self.final_criticality_exp = criticality_exp
        self.criticality_exp = 1.
        self.first_max_move_distance = max(*self.s2p.extent)
        self.delta_r = self.first_max_move_distance - 1.
        self.delta_e = self.final_criticality_exp - self.criticality_exp
        self.delta_e_over_delta_r = self.delta_e / self.delta_r
        self.criticality_const_term = (self.criticality_exp +
                                       (self.first_max_move_distance *
                                        self.delta_e_over_delta_r))
        self._arrival_times = None
        self._departure_times = None
        self.sync_block_keys = self._connections_table.sync_logic_block_keys()

    def update_state(self, maximum_move_distance):
        super(CAMIPTiming, self).update_state(maximum_move_distance)
        self.criticality_exp = (self.criticality_const_term -
                                maximum_move_distance *
                                self.delta_e_over_delta_r)
        self._arrival_times = self.arrival_times()
        self._departure_times = self.departure_times()
        self.critical_path = self._arrival_times[:].max()

        # Delay targets that have a synchronous logic block as a source _must_
        # treat the longest path delay of the source as 0!
        arrival_times = self._arrival_times[:]
        arrival_times[self.sync_block_keys] = 0
        self._arrival_times[:] = arrival_times
        departure_times = self._departure_times[:]
        departure_times[self.sync_block_keys] = 0
        self._departure_times[:] = departure_times

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

    #@profile
    #def run_iteration(self, seed, temperature, max_io_move=None,
                      #max_logic_move=None):

    @profile
    def evaluate_moves(self):
        super(CAMIPTiming, self).evaluate_moves()

        # If this is the first run iteration, we need to compute the arrival
        # times and departure times.  After the first run iteration, the times
        # will be updated whenever the `update_state` method is called.
        if self._departure_times is None:
            self.update_state(self.first_max_move_distance)

        # If timing-costs are disabled, we do not evaluate costs based on delay
        # information, so we skip and connection delay cost calculations.
        if self.timing_cost_disabled:
            return

        arch_data = self.departure_data.arch_data

        a = self.arrival_data.connections
        d = self.departure_data.connections
        look_up_delay_prime(a.v['source_key'], a.v['target_key'],
                            a.v['delay_type'], self.position_views['p_x'],
                            self.position_views['p_y'],
                            self.position_views['p_x_prime'],
                            self.position_views['p_y_prime'],
                            arch_data.v['delays'], arch_data.nrows,
                            arch_data.ncols, a.v['delay_prime'])

        look_up_delay_prime(d.v['source_key'], d.v['target_key'],
                            d.v['delay_type'], self.position_views['p_x'],
                            self.position_views['p_y'],
                            self.position_views['p_x_prime'],
                            self.position_views['p_y_prime'],
                            arch_data.v['delays'], arch_data.nrows,
                            arch_data.ncols, d.v['delay_prime'])

        block_arrays = DeviceDataFrame({'arrival_cost':
                                        np.zeros(self._arrival_times.size,
                                                 dtype=np.float32),
                                        'departure_cost':
                                        np.zeros(self._departure_times.size,
                                                 dtype=np.float32)})
        #block_arrays.v['arrival_cost'][:] = 0
        #block_arrays.v['departure_cost'][:] = 0

        connection_cost(self.criticality_exp, a.v['delay'],
                        self._arrival_times, self._departure_times,
                        a.v['source_key'], a.v['target_key'], a.v['cost'],
                        self.critical_path)
        connection_cost(self.criticality_exp, a.v['delay_prime'],
                        self._arrival_times, self._departure_times,
                        a.v['source_key'], a.v['target_key'],
                        a.v['cost_prime'], self.critical_path)

        connection_cost(self.criticality_exp, d.v['delay'],
                        self._arrival_times, self._departure_times,
                        d.v['target_key'], d.v['source_key'], d.v['cost'],
                        self.critical_path)
        connection_cost(self.criticality_exp, d.v['delay_prime'],
                        self._arrival_times, self._departure_times,
                        d.v['target_key'], d.v['source_key'],
                        d.v['cost_prime'], self.critical_path)

        #  - For now, we're just computing the cost of each block in terms of
        #    arriving paths and departing paths, based on `p_x` and `p_y`,
        #    _i.e., the current block positions_.
        #  - We're writing:
        #   * The sum of the cost of arriving connections for each block to the
        #     block-indexed position in `block_arrays.v['arrival_cost']`.
        #   * The sum of the cost of the departing connections for each block
        #     to the block-indexed position in
        #     `block_arrays.v['departure_cost']`.
        #
        # TODO
        # ====
        #
        #  - Compute the difference between `cost_prime` and `cost` before
        #    reducing by the target block key.  The resulting reduced delta
        #    values can then be scattered to the appropriate cost array, _i.e.,
        #    either the arrival cost array or departure cost array_.
        arrival_block_count, departure_block_count, max_timing_delta = \
            block_delta_timing_cost(
                a.v['target_key'], a.v['cost'], a.v['cost_prime'],
                d.v['target_key'], d.v['cost'], d.v['cost_prime'],
                a.v['reduced_keys'], d.v['reduced_keys'],
                a.v['reduced_target_cost'], d.v['reduced_target_cost'],
                block_arrays.v['arrival_cost'],
                block_arrays.v['departure_cost'])

        max_wirelength_delta = np.abs(self.delta_n[:]).max()
        delta_n_view = dv.view_from_vector(self.delta_n)
        compute_normalized_weighted_sum(self.wire_length_factor, delta_n_view,
                                        max_wirelength_delta,
                                        block_arrays.v['arrival_cost'],
                                        max_timing_delta, delta_n_view)

    def get_state(self):
        return {'critical_path': self.critical_path,
                'criticality_exp': self.criticality_exp}

    def exit_criteria(self, temperature):
        #return temperature < 0.00001 * self.theta / self.net_count
        return temperature < 0.005 / self.net_count
