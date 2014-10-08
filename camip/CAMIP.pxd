from libc.stdint cimport uint32_t, int32_t, uint8_t
from cythrust.thrust.pair cimport pair, make_pair
from cythrust.random cimport SimpleRNG, ParkMillerRNGBase


cdef extern from "math.h" nogil:
    double sqrt(double x)
    double exp(double x)
    double ceil(double x)


cdef extern from "std_dev.h" nogil:
    double c_get_std_dev "get_std_dev" (int n, double sum_x_squared, double av_x)


cdef extern from "CAMIP.hpp" nogil:
    cdef cppclass assess_group[T]:
        assess_group(T)

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


cdef extern from "camip_timing.h" nogil:
    cdef cppclass delay[T]:
        delay(T, int32_t, int32_t)

    cdef cppclass arrival_delay:
        pass


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


cdef class VPRAutoSlotKeyTo2dPosition:
    '''
    def __cinit__(self, int io_count, int logic_count, int io_capacity)
    '''
    cdef cVPRAutoSlotKeyTo2dPosition[pair[int32_t, int32_t]] *_data


cdef class VPRMovePattern:
    '''
    def __cinit__(int io_magnitude, int io_shift, logic_magnitude,
                  logic_shift,
                  VPRAutoSlotKeyTo2dPosition slot_key_to_position)
    '''
    cdef cVPRMovePattern[pair[int32_t, int32_t]] *_data


cdef class cAnnealSchedule:
    cdef AnnealSchedule *data
