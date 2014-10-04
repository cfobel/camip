#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t
from libc.math cimport fmin
import numpy as np
cimport numpy as np
from cythrust.thrust.pair cimport pair, make_pair
from cythrust.thrust.sort cimport sort_by_key
from cythrust.thrust.reduce cimport accumulate, accumulate_by_key, reduce_by_key
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.copy cimport copy_n
from cythrust.thrust.transform cimport transform, transform2
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport make_tuple5, make_tuple4, make_tuple2
from cythrust.thrust.functional cimport (unpack_binary_args, square, equal_to,
                                         unpack_quinary_args, plus, minus,
                                         reduce_plus4)
cimport cython

cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    double ceil(double x)


cdef extern from "math.h" nogil:
    double c_get_std_dev "get_std_dev" (int n, double sum_x_squared, double av_x)


cdef extern from "CAMIP.hpp" nogil:
    cdef cppclass evaluate_move:
        evaluate_move(float)

    cdef cppclass slot_move[T]:
        slot_move(T)

    cdef cppclass IOSegmentStarts[T]:
        IOSegmentStarts(T)

    cdef cppclass BlockTypeCount:
        int io
        int logic

    cdef cppclass VPRIOSlotKeyTo2dPosition[T]:
        VPRIOSlotKeyTo2dPosition(T)

    cdef cppclass SlotKeyTo2dPosition[T]:
        SlotKeyTo2dPosition(T)

    cdef cppclass MovePattern:
        MovePattern(int, int)

    cdef cppclass MovePatternInBounds:
        MovePatternInBounds(int extent, int magnitude, int shift)

    cdef cppclass MovePatternInBounds2d[T]:
        MovePatternInBounds2d(T magnitude, T shift,
                              VPRAutoSlotKeyTo2dPosition[T]
                              slot_key_to_position)

    cdef cppclass cVPRAutoSlotKeyTo2dPosition 'VPRAutoSlotKeyTo2dPosition' [T]:
        int io_count
        int logic_count
        int io_capacity
        T extent
        VPRIOSlotKeyTo2dPosition[T] io_s2p
        SlotKeyTo2dPosition[T] logic_s2p
        BlockTypeCount slot_count

        cVPRAutoSlotKeyTo2dPosition(int io_count, int logic_count,
                                    int io_capacity)
        T operator() (int)

    cdef cppclass PatternParams:
        int32_t magnitude
        int32_t shift

    cdef cppclass PatternParams2D[T]:
        PatternParams2D()
        T magnitude
        T shift

    cdef cppclass cVPRMovePattern 'VPRMovePattern' [T]:
        cVPRAutoSlotKeyTo2dPosition[T] s2p
        int io_slot_count
        MovePatternInBounds io_pattern
        MovePatternInBounds2d[T] logic_pattern
        int32_t operator() (int)
        size_t io_slot_count()
        size_t logic_slot_count()
        size_t total_slot_count()

        cVPRMovePattern(int io_magnitude, int io_shift, T logic_magnitude,
                        T logic_shift,
                        cVPRAutoSlotKeyTo2dPosition[T] slot_key_to_position)

    cdef cppclass block_group_key[T]:
        block_group_key(T)

    cdef cppclass c_star_plus_2d 'star_plus_2d' [T]:
        c_star_plus_2d(float)


cdef class VPRMovePattern:
    '''
    def __cinit__(int io_magnitude, int io_shift, logic_magnitude,
                  logic_shift,
                  VPRAutoSlotKeyTo2dPosition slot_key_to_position)
    '''
    cdef cVPRMovePattern[pair[int32_t, int32_t]] *_data

    def __cinit__(self, int io_magnitude, int io_shift, logic_magnitude,
                  logic_shift,
                  VPRAutoSlotKeyTo2dPosition slot_key_to_position):
        cdef pair[int32_t, int32_t] _logic_magnitude
        cdef pair[int32_t, int32_t] _logic_shift

        _logic_magnitude = make_pair(<int32_t>logic_magnitude[0],
                                     <int32_t>logic_magnitude[1])
        _logic_shift = make_pair(<int32_t>logic_shift[0],
                                 <int32_t>logic_shift[1])
        cdef cVPRAutoSlotKeyTo2dPosition[pair[int32_t, int32_t]] *t = \
            (<VPRAutoSlotKeyTo2dPosition>slot_key_to_position)._data

        self._data = new cVPRMovePattern[pair[int32_t, int32_t]](
            io_magnitude, io_shift, _logic_magnitude, _logic_shift, deref(t))

    def __call__(self, int k):
        return self[k]

    def __getitem__(self, int k):
        cdef int32_t result
        result = deref(self._data)(k)
        return result

    def __dealloc__(self):
        del self._data

    property io_slot_count:
        def __get__(self):
            return self._data.io_slot_count()

    property logic_slot_count:
        def __get__(self):
            return self._data.logic_slot_count()

    property total_slot_count:
        def __get__(self):
            return self._data.total_slot_count()


cdef class VPRAutoSlotKeyTo2dPosition:
    '''
    def __cinit__(self, int io_count, int logic_count, int io_capacity)
    '''
    cdef cVPRAutoSlotKeyTo2dPosition[pair[int32_t, int32_t]] *_data

    def __cinit__(self, int io_count, int logic_count, int io_capacity):
        self._data = new cVPRAutoSlotKeyTo2dPosition[pair[int32_t, int32_t]](io_count, logic_count, io_capacity)

    def __call__(self, int k):
        return self[k]

    def __getitem__(self, int k):
        cdef pair[int32_t, int32_t] result
        result = deref(self._data)(k)
        return (<int32_t>result.first, <int32_t>result.second)

    def __dealloc__(self):
        del self._data

    property io_count:
        def __get__(self):
            return self._data.io_count

        def __set__(self, value):
            self._data.io_count = value

    property logic_count:
        def __get__(self):
            return self._data.logic_count

        def __set__(self, value):
            self._data.logic_count = value

    property io_capacity:
        def __get__(self):
            return self._data.io_capacity

        def __set__(self, value):
            self._data.io_capacity = value

    property extent:
        def __get__(self):
            return (<int32_t>self._data.extent.first, <int32_t>self._data.extent.second)

        def __set__(self, value):
            assert(len(value) == 2)
            cdef pair[int32_t, int32_t] result
            cdef int32_t first = int(value[0])
            cdef int32_t second = int(value[2])
            result = make_pair(first, second)
            self._data.extent = result

    property io_slot_count:
        def __get__(self):
            return self._data.slot_count.io

    property logic_slot_count:
        def __get__(self):
            return self._data.slot_count.logic

    property total_slot_count:
        def __get__(self):
            return self._data.slot_count.io + self._data.slot_count.logic


cdef extern from "schedule.hpp" nogil:
    cppclass AnnealSchedule "anneal::AnnealSchedule<float>":
        float start_rlim_
        float rlim_
        float start_temperature_
        float temperature_
        float success_ratio_

        AnnealSchedule(float start_rlim, float start_temperature)
        int clamp_rlim(float max_rlim)
        int get_temperature_stage() const
        void init(float start_rlim, float start_temperature)
        void update_rlim()
        void update_state(float success_ratio)
        void update_temperature()


cpdef sum_xy_vectors(int32_t[:] block_keys, int32_t[:] net_keys,
                     int32_t[:] p_x, int32_t[:] p_y,
                     float[:] e_x, float[:] e_x2,
                     float[:] e_y, float[:] e_y2,
                     int32_t[:] reduced_keys):
    cdef size_t count = net_keys.size
    cdef square[float] square_f
    cdef equal_to[int32_t] reduce_compare
    cdef reduce_plus4[float] reduce_plus4

    reduce_by_key(
        &net_keys[0],  # `keys_first`
        &net_keys[0] + count,  # `keys_last`
        make_zip_iterator(  # `values_first`
            make_tuple4(
                make_permutation_iterator(&p_x[0], &block_keys[0]),
                make_transform_iterator(
                    make_permutation_iterator(&p_x[0], &block_keys[0]), square_f),
                make_permutation_iterator(&p_y[0], &block_keys[0]),
                make_transform_iterator(
                    make_permutation_iterator(&p_y[0], &block_keys[0]), square_f))),
        &reduced_keys[0],  # `keys_output`
        make_zip_iterator(make_tuple4(&e_x[0], &e_x2[0], &e_y[0], &e_y2[0])),
        reduce_compare, reduce_plus4)


cpdef copy_e_c_to_omega(float[:] e_c, int32_t[:] block_keys, float[:] omega):
    copy_n(
        make_permutation_iterator(&e_c[0], &block_keys[0]),
        <size_t>block_keys.size, &omega[0])


cpdef sort_float_coo(int32_t[:] keys1, int32_t[:] keys2, float[:] values):
    sort_by_key(&keys1[0], &keys1[0] + <size_t>keys1.size,
                make_zip_iterator(make_tuple2(&keys2[0], &values[0])))


cpdef sort_netlist_keys(int32_t[:] keys1, int32_t[:] keys2):
    sort_by_key(&keys1[0], &keys1[0] + <size_t>keys1.size, &keys2[0])


cpdef sum_float_by_key(int32_t[:] keys, float[:] values,
                       int32_t[:] reduced_keys, float[:] reduced_values):
    cdef size_t count = (<int32_t*>accumulate_by_key(&keys[0], &keys[0] +
                                                     <size_t>keys.size,
                                                     &values[0],
                                                     &reduced_keys[0],
                                                     &reduced_values[0]).first
                         - &reduced_keys[0])
    return count


cpdef evaluate_moves(int32_t[:] row, int32_t[:] col, int32_t[:] p_x,
                     int32_t[:] p_x_prime, float[:] e_x, float[:] e_x_2,
                     int32_t[:] p_y, int32_t[:] p_y_prime, float[:] e_y,
                     float[:] e_y_2, float[:] r_inv, float beta,
                     int32_t[:] reduced_keys, float[:] reduced_values):
    cdef size_t count = <size_t>reduced_values.size
    cdef int k
    cdef int i
    cdef int j

    cdef plus[float] plus2
    cdef unpack_binary_args[plus[float]] *plus2_tuple = \
        new unpack_binary_args[plus[float]](plus2)
    cdef evaluate_move *eval_func = new evaluate_move(beta)
    cdef unpack_quinary_args[evaluate_move] *eval_func_tuple = \
        new unpack_quinary_args[evaluate_move](deref(eval_func))

    accumulate_by_key(
        &row[0], &row[0] + count,
        make_transform_iterator(
            make_zip_iterator(
                make_tuple2(
                    make_transform_iterator(
                        make_zip_iterator(
                            make_tuple5(
                                make_permutation_iterator(&e_y[0], &col[0]),
                                make_permutation_iterator(&e_y_2[0], &col[0]),
                                make_permutation_iterator(&p_y[0], &row[0]),
                                make_permutation_iterator(&p_y_prime[0], &row[0]),
                                make_permutation_iterator(&r_inv[0], &col[0]))),
                        deref(eval_func_tuple)),
                    make_transform_iterator(
                        make_zip_iterator(
                            make_tuple5(
                                make_permutation_iterator(&e_x[0], &col[0]),
                                make_permutation_iterator(&e_x_2[0], &col[0]),
                                make_permutation_iterator(&p_x[0], &row[0]),
                                make_permutation_iterator(&p_x_prime[0], &row[0]),
                                make_permutation_iterator(&r_inv[0], &col[0]))),
                        deref(eval_func_tuple)))), deref(plus2_tuple)),
        &reduced_keys[0], &reduced_values[0])


cdef class cAnnealSchedule:
    cdef AnnealSchedule *data

    def __cinit__(self, float start_rlim=0, float start_temperature=0):
        self.data = new AnnealSchedule(start_rlim, start_temperature)

    def __dealloc__(self):
        del self.data

    property start_rlim:
        def __get__(self):
            return self.data.start_rlim_
        def __set__(self, value):
            self.data.start_rlim_ = value

    property rlim:
        def __get__(self):
            return self.data.rlim_
        def __set__(self, value):
            self.data.rlim_ = value

    property start_temperature:
        def __get__(self):
            return self.data.start_temperature_
        def __set__(self, value):
            self.data.start_temperature_ = value

    property temperature:
        def __get__(self):
            return self.data.temperature_
        def __set__(self, value):
            self.data.temperature_ = value

    property success_ratio:
        def __get__(self):
            return self.data.success_ratio_
        def __set__(self, value):
            self.data.success_ratio_ = value

    def clamp_rlim(self, max_rlim):
        return self.data.clamp_rlim(max_rlim)

    def get_temperature_stage(self):
        return self.data.get_temperature_stage()

    def init(self, start_rlim, start_temperature):
        self.data.init(start_rlim, start_temperature)

    def update_rlim(self):
        self.data.update_rlim()

    def update_state(self, success_ratio):
        self.data.update_state(success_ratio)

    def update_temperature(self):
        self.data.update_temperature()


cdef PatternParams2D[pair[int32_t, int32_t]] random_2d_pattern_params(
        pair[int32_t,int32_t] max_magnitude, VPRAutoSlotKeyTo2dPosition vpr_s2p):
    cdef pair[int32_t, int32_t] m
    m = make_pair(<int32_t>fmin(<int32_t>vpr_s2p._data.extent.first - 1,
                                <int32_t>max_magnitude.first),
                  <int32_t>fmin(<int32_t>vpr_s2p._data.extent.second - 1,
                                <int32_t>max_magnitude.second))

    cdef PatternParams2D[pair[int32_t, int32_t]] params

    params.magnitude = make_pair(
        <int32_t>np.random.randint(0, <int32_t>max_magnitude.first + 1),
        <int32_t>np.random.randint(0, <int32_t>max_magnitude.second + 1))

    while ((<int32_t>params.magnitude.first == 0) and
           (<int32_t>params.magnitude.second == 0)):
        params.magnitude = make_pair(
            <int32_t>np.random.randint(0, <int32_t>max_magnitude.first + 1),
            <int32_t>np.random.randint(0, <int32_t>max_magnitude.second + 1))

    cdef int32_t max_shift
    cdef int32_t shift_first = 0
    cdef int32_t shift_second = 0

    if <int32_t>params.magnitude.first > 0:
        max_shift = 2 * <int32_t>params.magnitude.first - 1
        if <int32_t>vpr_s2p._data.extent.first <= 2 * <int32_t>params.magnitude.first:
            max_shift = <int32_t>params.magnitude.first - 1
        shift_first = np.random.randint(max_shift + 1)

    if <int32_t>params.magnitude.second > 0:
        max_shift = 2 * <int32_t>params.magnitude.second - 1
        if <int32_t>(vpr_s2p._data.extent).second <= (2 * <int32_t>params.magnitude.second):
            max_shift = <int32_t>params.magnitude.second - 1
        shift_second = np.random.randint(max_shift + 1)
    params.shift = make_pair(shift_first, shift_second)
    return params


cdef PatternParams random_pattern_params(int32_t max_magnitude, int32_t extent):
    max_magnitude = <int32_t>fmin(extent - 1, max_magnitude)
    cdef int32_t magnitude = <int32_t>np.random.randint(0, max_magnitude + 1)

    while magnitude == 0:
        magnitude = <int32_t>np.random.randint(0, max_magnitude + 1)

    shift = 0

    if magnitude > 0:
        max_shift = 2 * magnitude - 1
        if extent <= 2 * magnitude:
            max_shift = magnitude - 1
        shift = <int32_t>np.random.randint(max_shift + 1)

    cdef PatternParams result
    result.magnitude = magnitude
    result.shift = shift
    return result


def random_vpr_pattern(VPRAutoSlotKeyTo2dPosition vpr_s2p, max_logic_move=None,
                       max_io_move=None):
    cdef int32_t io_extent = vpr_s2p._data.slot_count.io
    cdef PatternParams io_params
    cdef PatternParams2D[pair[int32_t, int32_t]] logic_params
    cdef pair[int32_t, int32_t] _max_logic_move

    if max_io_move is None:
        max_io_move = io_extent - 1
    io_params = random_pattern_params(max_io_move, io_extent)
    if max_logic_move is None:
        _max_logic_move = make_pair(<int32_t>(<int32_t>vpr_s2p._data.extent.first - 1),
                                    <int32_t>(<int32_t>vpr_s2p._data.extent.second - 1))
    else:
        _max_logic_move = make_pair(<int32_t>max_logic_move[0],
                                    <int32_t>max_logic_move[1])
    logic_params = random_2d_pattern_params(_max_logic_move, vpr_s2p)
    return VPRMovePattern(io_params.magnitude, io_params.shift,
                          (<int32_t>logic_params.magnitude.first,
                           <int32_t>logic_params.magnitude.second),
                          (<int32_t>logic_params.shift.first,
                           <int32_t>logic_params.shift.second), vpr_s2p)


cpdef slot_moves(uint32_t[:] slot_keys, uint32_t[:] slot_keys_prime,
                 VPRMovePattern move_pattern):
    cdef size_t count = <size_t>slot_keys.size
    cdef plus[uint32_t] plus2_func

    transform2(&slot_keys[0], &slot_keys[0] + count,
               make_transform_iterator(&slot_keys[0],
                                       deref(move_pattern._data)),
               &slot_keys_prime[0], plus2_func)


cpdef extract_positions(int32_t[:] p_x, int32_t[:] p_y, uint32_t[:] slot_keys,
                        VPRAutoSlotKeyTo2dPosition s2p):
    # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    # slot assignments.
    cdef int i
    cdef pair[int32_t, int32_t] position

    for i in xrange(len(slot_keys)):
        position = deref(s2p._data)(slot_keys[i])
        p_x[i] = <int32_t>position.first
        p_y[i] = <int32_t>position.second


def get_std_dev(int n, double sum_x_squared, double av_x):
    return c_get_std_dev(n, sum_x_squared, av_x)


def compute_block_group_keys(uint32_t[:] block_slot_keys,
                             uint32_t[:] block_slot_keys_prime,
                             int32_t[:] block_group_keys,
                             int32_t sentinel_key):
    cdef size_t count = block_slot_keys.size
    cdef block_group_key[int32_t] *group_key_func = \
        new block_group_key[int32_t](sentinel_key)

    transform2(&block_slot_keys[0], &block_slot_keys[0] + count,
               &block_slot_keys_prime[0], &block_group_keys[0],
               deref(group_key_func))


def compute_move_deltas(float[:] n_c, float[:] n_c_prime, float[:] delta_n):
    cdef size_t count = n_c.size
    cdef minus[float] minus_func

    transform2(&n_c[0], &n_c[0] + count, &n_c_prime[0], &delta_n[0],
               minus_func)


cpdef sum_permuted_float_by_key(int32_t[:] keys, float[:] elements,
                                int32_t[:] index, int32_t[:] reduced_keys,
                                float[:] reduced_values):
    cdef size_t count = <int32_t*>accumulate_by_key(
        &keys[0], &keys[0] + <size_t>keys.size,
        make_permutation_iterator(&elements[0], &index[0]),
        &reduced_keys[0], &reduced_values[0]).first - &reduced_keys[0]
    return count


cpdef star_plus_2d(float[:] e_x, float[:] e_x2, float[:] e_y, float[:] e_y2,
                   float[:] r_inv, float beta, float[:] e_c):
    cdef size_t count = e_x.size

    cdef c_star_plus_2d[float] *_star_plus = new c_star_plus_2d[float](beta)
    cdef unpack_quinary_args[c_star_plus_2d[float]] *_star_plus_2d = \
        new unpack_quinary_args[c_star_plus_2d[float]](deref(_star_plus))

    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple5(&e_x[0], &e_x2[0], &e_y[0], &e_y2[0], &r_inv[0])),
            deref(_star_plus_2d)), count, &e_c[0])
    return <float>accumulate(&e_c[0], &e_c[0] + count)
