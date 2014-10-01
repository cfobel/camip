#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t
import numpy as np
cimport numpy as np
from cythrust.thrust.sort cimport sort_by_key
from cythrust.thrust.reduce cimport accumulate_by_key
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.copy cimport copy_n
from cythrust.thrust.transform cimport transform
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport make_tuple5, make_tuple2
from cythrust.thrust.functional cimport (unpack_binary_args, square,
                                         unpack_quinary_args, plus)
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)
    double ceil(double x)


cdef extern from "math.h":
    double c_get_std_dev "get_std_dev" (int n, double sum_x_squared, double av_x)


cdef extern from "CAMIP.hpp":
    cdef cppclass evaluate_move:
        evaluate_move(float)


cdef extern from "schedule.hpp":
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


cpdef sum_xy_vectors(int32_t[:] net_keys, float[:] X, float[:] Y,
                     float[:] e_x, float[:] e_x2,
                     float[:] e_y, float[:] e_y2,
                     int32_t[:] reduced_keys):
    cdef size_t count = net_keys.size
    cdef square[float] square_f

    accumulate_by_key(
        &net_keys[0], &net_keys[0] + count,
        &X[0], &reduced_keys[0], &e_x[0])
    accumulate_by_key(
        &net_keys[0], &net_keys[0] + count,
        make_transform_iterator(&X[0], square_f), &reduced_keys[0], &e_x2[0])
    accumulate_by_key(
        &net_keys[0], &net_keys[0] + count,
        &Y[0], &reduced_keys[0], &e_y[0])
    accumulate_by_key(
        &net_keys[0], &net_keys[0] + count,
        make_transform_iterator(&Y[0], square_f), &reduced_keys[0], &e_y2[0])


cpdef copy_e_c_to_omega(float[:] e_c, int32_t[:] block_keys, float[:] omega):
    copy_n(
        make_permutation_iterator(&e_c[0], &block_keys[0]),
        <size_t>block_keys.size, &omega[0])


cpdef sort_float_coo(int32_t[:] keys1, int32_t[:] keys2, float[:] values):
    sort_by_key(&keys1[0], &keys1[0] + <size_t>keys1.size,
                make_zip_iterator(make_tuple2(&keys2[0], &values[0])))


cpdef sort_netlist_keys(int32_t[:] keys1, int32_t[:] keys2):
    sort_by_key(&keys1[0], &keys1[0] + <size_t>keys1.size, &keys2[0])


cpdef sum_float_by_key(np.ndarray[np.int32_t,ndim=1] keys,
                       np.ndarray[np.float32_t,ndim=1] values,
                       np.ndarray[np.int32_t,ndim=1] reduced_keys,
                       np.ndarray[np.float32_t,ndim=1] reduced_values):
    cdef size_t count = (<int32_t*>accumulate_by_key(<int32_t *>keys.data,
                                                     <int32_t *>keys.data +
                                                     <size_t>keys.size,
                                                     <float *>values.data,
                                                     <int32_t *>reduced_keys.data,
                                                     <float *>reduced_values.data).first -
                         <int32_t*>reduced_keys.data)
    return count


cpdef evaluate_moves(float[:] output, int32_t[:] row, int32_t[:] col,
                     int32_t[:] p_x, int32_t[:] p_x_prime,
                     float[:] e_x, float[:] e_x_2,
                     int32_t[:] p_y, int32_t[:] p_y_prime,
                     float[:] e_y, float[:] e_y_2,
                     float[:] r_inv, float beta):
    cdef size_t count = len(output)
    cdef int k
    cdef int i
    cdef int j

    cdef plus[float] plus2
    cdef unpack_binary_args[plus[float]] *plus2_tuple = \
        new unpack_binary_args[plus[float]](plus2)
    cdef evaluate_move *eval_func = new evaluate_move(beta)
    cdef unpack_quinary_args[evaluate_move] *eval_func_tuple = \
        new unpack_quinary_args[evaluate_move](deref(eval_func))

    copy_n(
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
                        deref(eval_func_tuple)))), deref(plus2_tuple)), count,
        &output[0])

    # for k in xrange(count):
    #     i = row[k]
    #     j = col[k]
    #     output[k] = beta * (evaluate_move(i, j, p_x, p_x_prime, e_x, e_x_2,
    #                                       r_inv) +
    #                         evaluate_move(i, j, p_y, p_y_prime, e_y, e_y_2,
    #                                       r_inv))


cdef class BlockTypeCount:
    cdef int io
    cdef int logic

    def __cinit__(self, int io=0, int logic=0):
        self.io = io
        self.logic = logic

    def __add__(self, other):
        return (self.io + other.io, self.logic + other.logic)

    def __sub__(self, other):
        return (self.io - other.io, self.logic - other.logic)

    property io:
        def __get__(self):
            return self.io

        def __set__(self, value):
            self.io = value

    property logic:
        def __get__(self):
            return self.logic

        def __set__(self, value):
            self.logic = value


cdef class IOSegmentStarts:
    cdef int bottom
    cdef int right
    cdef int top
    cdef int left

    def __cinit__(self, Extent2D io_count):
        self.bottom = 0
        self.right = io_count.row
        self.top = self.right + io_count.column
        self.left = self.top + io_count.row

    property bottom:
        def __get__(self):
            return self.bottom

    property right:
        def __get__(self):
            return self.right

    property top:
        def __get__(self):
            return self.top

    property left:
        def __get__(self):
            return self.left


cdef class Extent2D:
    cdef int row
    cdef int column

    def __cinit__(self, int row=0, int column=0):
        self.row = row
        self.column = column

    def __add__(self, other):
        return (self.row + other.row, self.column + other.column)

    property row:
        def __get__(self):
            return self.row

        def __set__(self, value):
            self.row = value

    property column:
        def __get__(self):
            return self.column

        def __set__(self, value):
            self.column = value


cdef class TwoD:
    cdef int x
    cdef int y

    def __cinit__(self, int x=0, int y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return TwoD(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return TwoD(self.x - other.x, self.y - other.y)

    property x:
        def __get__(self):
            return self.x

        def __set__(self, value):
            self.x = value

    property y:
        def __get__(self):
            return self.y

        def __set__(self, value):
            self.y = value


cdef class SlotKeyTo2dPosition:
    cdef int row_extent
    cdef TwoD offset

    def __cinit__(self, int row_extent, TwoD offset=None):
        if offset is None:
            self.offset = TwoD(0, 0)
        else:
            self.offset = offset
        self.row_extent = row_extent

    def __getitem__(self, k):
        return self.get(k)

    cdef get(self, int k):
        cdef TwoD p = TwoD(k // self.row_extent, k % self.row_extent)
        return p + self.offset

    property row_extent:
        def __get__(self):
            return self.row_extent

    property offset:
        def __get__(self):
            return self.offset


cdef class VPRIOSlotKeyTo2dPosition:
    cdef Extent2D extent
    cdef int io_capacity
    cdef IOSegmentStarts segment_start
    cdef int size

    def __cinit__(self, Extent2D extent, int io_capacity):
        self.extent = extent
        self.io_capacity = io_capacity

        cdef Extent2D io_count = Extent2D(self.extent.row * self.io_capacity,
                                          self.extent.column *
                                          self.io_capacity)
        self.segment_start = IOSegmentStarts(io_count)
        self.size = 2 * (io_count.column + io_count.row)

    property extent:
        def __get__(self):
            return self.extent

    property io_capacity:
        def __get__(self):
            return self.io_capacity

    property segment_start:
        def __get__(self):
            return self.segment_start

    def __len__(self):
        return self.size

    def __getitem__(self, k):
        return self.get(k)

    cdef get(self, int k):
        if k < self.segment_start.right:
            # Slot `k` maps to position along the 'bottom' of the grid.
            return TwoD(0, 1 + k // self.io_capacity)
        elif k < self.segment_start.top:
            # Slot `k` maps to position along the ['right'] side of the grid.
            return TwoD(1 + (k - self.segment_start.right) // self.io_capacity,
                        self.extent.row + 1)
        elif k < self.segment_start.left:
            # Slot `k` maps to position along the top of the grid.
            return TwoD(self.extent.column + 1,
                        self.extent.row - (k - self.segment_start.top)
                        // self.io_capacity)
        else:
            # Assume slot `k` maps to position along the left of the grid.
            return TwoD(self.extent.column - (k - self.segment_start.left) //
                        self.io_capacity, 0)


cdef class VPRAutoSlotKeyTo2dPosition:
    cdef int io_count
    cdef int logic_count
    cdef int io_capacity
    cdef Extent2D extent
    cdef VPRIOSlotKeyTo2dPosition io_s2p
    cdef SlotKeyTo2dPosition logic_s2p
    cdef BlockTypeCount slot_count

    def __cinit__(self, int io_count, int logic_count, int io_capacity=2):
        self.io_count = io_count
        self.logic_count = logic_count
        self.io_capacity = io_capacity

        self.extent = Extent2D()
        self.extent.row = <int>ceil(sqrt(logic_count))
        self.extent.column = self.extent.row

        if (io_count > 2 * self.io_capacity * (self.extent.row +
                                               self.extent.column)):
            # The size determined based on the number of logic blocks does not
            # provide enough spots for the inputs/outputs along the perimeter.
            # Increase extents of the grid to fit IO.
            self.extent.row = <int>ceil(sqrt(io_count + logic_count))
            self.extent.column = self.extent.row

        cdef Extent2D io_extent

        if (io_count > 0):
            io_extent = self.extent

        self.io_s2p = VPRIOSlotKeyTo2dPosition(io_extent, self.io_capacity)

        # Logic grid starts at `(1, 1)`.
        self.logic_s2p = SlotKeyTo2dPosition(self.extent.row, TwoD(1, 1))
        self.slot_count = BlockTypeCount(len(self.io_s2p), self.extent.row *
                                         self.extent.column)

    def __len__(self):
        return self.slot_count.io + self.slot_count.logic

    cdef inline position0(self, TwoD position):
        return position - self.logic_s2p.offset

    cdef inline in_bounds(self, TwoD position):
        cdef TwoD p = self.position0(position)
        return not (p.x < 0 or p.x >= self.extent.column or p.y < 0
                    or p.y >= self.extent.row)

    def __getitem__(self, k):
        return self.get(k)

    cdef get0(self, k):
        return self.position0(self.get(k))

    cdef get(self, k):
        if k < self.slot_count.io:
            return self.io_s2p.get(k)
        else:
            return self.logic_s2p.get(k - self.slot_count.io)

    property slot_count:
        def __get__(self):
            return self.slot_count

    property io_s2p:
        def __get__(self):
            return self.io_s2p

    property logic_s2p:
        def __get__(self):
            return self.logic_s2p

    property extent:
        def __get__(self):
            return self.extent

    property io_capacity:
        def __get__(self):
            return self.io_capacity


cdef class MovePattern:
    cdef int magnitude
    cdef int double_magnitude
    cdef int shift

    def __cinit__(self, int magnitude, int shift=0):
        self.magnitude = magnitude
        self.double_magnitude = 2 * magnitude
        self.shift = shift

    def __getitem__(self, int i):
        return self.get(i)

    cdef inline get(self, int i):
        if self.magnitude == 0:
            return 0
        cdef int index = (i + 2 * self.double_magnitude -
                          ((self.shift + self.magnitude + 1) %
                           self.double_magnitude))
        if (index % self.double_magnitude) < self.magnitude:
            return self.magnitude
        else:
            return -self.magnitude

    property magnitude:
        def __get__(self):
            return self.magnitude

        def __set__(self, value):
            self.magnitude = value
            self.double_magnitude = 2 * self.magnitude

    property shift:
        def __get__(self):
            return self.shift

        def __set__(self, value):
            self.shift = value


cdef class MovePattern2d:
    cdef Extent2D extent
    cdef int size
    cdef MovePattern row
    cdef MovePattern column

    def __init__(self, Extent2D magnitude, Extent2D shift, Extent2D extent):
        self.row = MovePattern(magnitude.row, shift.row)
        self.column = MovePattern(magnitude.column, shift.column)
        self.extent = extent
        self.size = extent.row * extent.column

    cdef inline column_i(self, int i):
        return ((i // self.extent.row) + self.extent.column *
                (i % self.extent.row))

    cdef inline get_column(self, int i):
        return self.column.get(self.column_i(i))

    cdef inline get_row(self, int i):
        return self.row.get(i)

    cdef inline get(self, int i):
        return Extent2D(self.get_row(i), self.get_column(i))

    def __getitem__(self, int i):
        return self.get(i)

    def __len__(self):
        return self.size

    property row:
        def __get__(self):
            return self.row

    property column:
        def __get__(self):
            return self.column


cdef class MovePatternInBounds:
    cdef int extent
    cdef MovePattern pattern

    def __init__(self, int extent, int magnitude, int shift=0):
        self.extent = extent
        self.pattern = MovePattern(magnitude, shift)

    def __getitem__(self, int i):
        return self.get(i)

    cdef inline int get(self, int i):
        # We still need to use the offset-based position for computing the
        # target position.
        cdef int move = self.pattern.get(i)
        cdef int target = i + move
        if target < 0 or target >= self.extent:
            # If the displacement targets a location that is outside the
            # boundaries, set displacement to zero.
            return 0
        return move

    def __len__(self):
        return self.extent

    property extent:
        def __get__(self): return self.extent

    property magnitude:
        def __get__(self):
            return self.pattern.magnitude
        def __set__(self, value):
            self.pattern.magnitude = value

    property shift:
        def __get__(self):
            return self.pattern.shift
        def __set__(self, value):
            self.pattern.shift = value

    property pattern:
        def __get__(self): return self.pattern


cdef class MovePatternInBounds2d:
    cdef VPRAutoSlotKeyTo2dPosition s2p
    cdef MovePattern2d pattern

    def __cinit__(self, Extent2D magnitude, Extent2D shift,
                  VPRAutoSlotKeyTo2dPosition slot_key_to_position):
        self.s2p = slot_key_to_position
        self.pattern = MovePattern2d(magnitude, shift, self.s2p.extent)

    def __getitem__(self, i):
        return self.get(i)

    cdef inline int get(self, int i):
        # Get zero-based position, since displacement patterns are indexed
        # starting at zero.
        cdef TwoD position0 = self.s2p.get0(i)

        # We still need to use the offset-based position for computing the
        # target position.
        cdef TwoD position = self.s2p.get(i)
        cdef TwoD move = TwoD(self.pattern.column.get(position0.x),
                              self.pattern.row.get(position0.y))
        cdef TwoD target = position + move
        if not self.s2p.in_bounds(target):
            # If the displacement targets a location that is outside the
            # boundaries, set displacement to zero.
            return 0
        return move.x * self.s2p.extent.row + move.y

    def __len__(self):
        return len(self.pattern)

    property pattern:
        def __get__(self):
            return self.pattern

    property magnitude:
        def __get__(self):
            return self.pattern.magnitude
        def __set__(self, value):
            self.pattern.magnitude = value

    property shift:
        def __get__(self):
            return self.pattern.shift
        def __set__(self, value):
            self.pattern.shift = value


cdef class VPRMovePattern:
    cdef VPRAutoSlotKeyTo2dPosition s2p
    cdef int io_slot_count
    cdef MovePatternInBounds io_pattern
    cdef MovePatternInBounds2d logic_pattern

    def __init__(self, int io_magnitude, int io_shift,
                 Extent2D logic_magnitude, Extent2D logic_shift,
                 VPRAutoSlotKeyTo2dPosition slot_key_to_position):
        self.s2p = slot_key_to_position
        self.io_slot_count = self.s2p.slot_count.io
        self.io_pattern = MovePatternInBounds(self.io_slot_count,
                                              io_magnitude, io_shift)
        self.logic_pattern = MovePatternInBounds2d(logic_magnitude,
                                                   logic_shift, self.s2p)

    def __getitem__(self, i):
        return self.get(i)

    cdef inline int get(self, int i):
        if i < self.io_slot_count:
            return self.io_pattern.get(i)
        else:
            return self.logic_pattern.get(i)

    def __len__(self):
        return len(self.s2p)

    property io_pattern:
        def __get__(self): return self.io_pattern

    property logic_pattern:
        def __get__(self): return self.logic_pattern


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


cpdef slot_moves(int32_t[:] output, uint32_t[:] slot_keys,
                 VPRMovePattern move_pattern):
    cdef int i

    for i in xrange(len(output)):
        output[i] = move_pattern.get(slot_keys[i])


cpdef extract_positions(int32_t[:] p_x, int32_t[:] p_y, uint32_t[:] slot_keys,
                        VPRAutoSlotKeyTo2dPosition s2p):
    # Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    # slot assignments.
    cdef int i
    cdef TwoD position

    for i in xrange(len(slot_keys)):
        position = s2p.get(slot_keys[i])
        p_x[i] = position.x
        p_y[i] = position.y


def get_std_dev(int n, double sum_x_squared, double av_x):
    return c_get_std_dev(n, sum_x_squared, av_x)
