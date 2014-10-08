#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t, uint8_t, int8_t
from libc.math cimport fmin
import numpy as np
cimport numpy as np

from cythrust.device_vector cimport (DeviceVectorInt32, DeviceVectorUint32,
                                     DeviceVectorFloat32, DeviceVectorInt8)
from cythrust.thrust.copy cimport copy_n, copy_if_w_stencil, copy
from cythrust.thrust.device_vector cimport device_vector
from cythrust.thrust.fill cimport fill_n
from cythrust.thrust.functional cimport (unpack_binary_args, square, equal_to,
                                         not_equal_to, unpack_quinary_args,
                                         plus, minus, reduce_plus4, identity,
                                         logical_not, absolute,
                                         unpack_ternary_args)
from cythrust.thrust.iterator.counting_iterator cimport counting_iterator
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.repeated_range_iterator cimport repeated_range
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.reduce cimport accumulate, accumulate_by_key, reduce_by_key
from cythrust.thrust.scan cimport exclusive_scan, inclusive_scan
from cythrust.thrust.sequence cimport sequence
from cythrust.thrust.sort cimport sort_by_key, sort
from cythrust.thrust.transform cimport transform, transform2
from cythrust.thrust.tuple cimport (make_tuple2, make_tuple3, make_tuple4,
                                    make_tuple5)
from camip.CAMIP cimport (VPRAutoSlotKeyTo2dPosition, evaluate_move,
                          VPRMovePattern, c_star_plus_2d, block_group_key,
                          assess_group, delay)


cpdef evaluate_moves(DeviceVectorInt32 row, DeviceVectorInt32 col,
                     DeviceVectorInt32 p_x, DeviceVectorInt32 p_x_prime,
                     DeviceVectorFloat32 e_x, DeviceVectorFloat32 e_x_2,
                     DeviceVectorInt32 p_y, DeviceVectorInt32 p_y_prime,
                     DeviceVectorFloat32 e_y, DeviceVectorFloat32 e_y_2,
                     DeviceVectorFloat32 r_inv, float beta,
                     DeviceVectorInt32 reduced_keys,
                     DeviceVectorFloat32 reduced_values):
    cdef plus[float] plus2
    cdef unpack_binary_args[plus[float]] *plus2_tuple = \
        new unpack_binary_args[plus[float]](plus2)
    cdef evaluate_move *eval_func = new evaluate_move(beta)
    cdef unpack_quinary_args[evaluate_move] *eval_func_tuple = \
        new unpack_quinary_args[evaluate_move](deref(eval_func))

    accumulate_by_key(
        row._vector.begin(), row._vector.end(),
        make_transform_iterator(
            make_zip_iterator(
                make_tuple2(
                    make_transform_iterator(
                        make_zip_iterator(
                            make_tuple5(
                                make_permutation_iterator(e_y._vector.begin(), col._vector.begin()),
                                make_permutation_iterator(e_y_2._vector.begin(), col._vector.begin()),
                                make_permutation_iterator(p_y._vector.begin(), row._vector.begin()),
                                make_permutation_iterator(p_y_prime._vector.begin(), row._vector.begin()),
                                make_permutation_iterator(r_inv._vector.begin(), col._vector.begin()))),
                        deref(eval_func_tuple)),
                    make_transform_iterator(
                        make_zip_iterator(
                            make_tuple5(
                                make_permutation_iterator(e_x._vector.begin(), col._vector.begin()),
                                make_permutation_iterator(e_x_2._vector.begin(), col._vector.begin()),
                                make_permutation_iterator(p_x._vector.begin(), row._vector.begin()),
                                make_permutation_iterator(p_x_prime._vector.begin(), row._vector.begin()),
                                make_permutation_iterator(r_inv._vector.begin(), col._vector.begin()))),
                        deref(eval_func_tuple)))), deref(plus2_tuple)),
        reduced_keys._vector.begin(), reduced_values._vector.begin())
    del plus2_tuple
    del eval_func
    del eval_func_tuple


def minus_float(DeviceVectorFloat32 n_c, DeviceVectorFloat32 n_c_prime,
                DeviceVectorFloat32 delta_n):
    cdef size_t count = delta_n.size
    cdef minus[float] minus_func

    transform2(n_c._vector.begin(), n_c._vector.begin() + count,
               n_c_prime._vector.begin(), delta_n._vector.begin(), minus_func)


cpdef slot_moves(DeviceVectorUint32 slot_keys, DeviceVectorUint32 slot_keys_prime,
                 VPRMovePattern move_pattern):
    cdef size_t count = <size_t>slot_keys.size
    cdef plus[uint32_t] plus2_func

    transform2(slot_keys._vector.begin(), slot_keys._vector.begin() + count,
               make_transform_iterator(slot_keys._vector.begin(),
                                       deref(move_pattern._data)),
               slot_keys_prime._vector.begin(), plus2_func)


cpdef extract_positions(DeviceVectorUint32 slot_keys, DeviceVectorInt32 p_x,
                        DeviceVectorInt32 p_y, VPRAutoSlotKeyTo2dPosition s2p):
    r'''
    Extract positions into $\vec{p_x}$ and $\vec{p_x}$ based on permutation
    slot assignments.
    '''
    cdef size_t count = slot_keys.size
    transform(slot_keys._vector.begin(), slot_keys._vector.begin() + count,
              make_zip_iterator(make_tuple2(p_x._vector.begin(),
                                            p_y._vector.begin())),
              deref(s2p._data))


def copy_int32(DeviceVectorInt32 a, DeviceVectorInt32 b):
    '''
    Equivalent to:

        b[:] = a[:]
    '''
    copy(a._vector.begin(), a._vector.end(), b._vector.begin())


def copy_permuted_uint32(DeviceVectorUint32 a, DeviceVectorUint32 b,
                         DeviceVectorInt32 index, size_t index_count):
    '''
    Equivalent to:

        b[index] = a[index]

    where index is an array of indexes corresponding to positions to copy from
    `a` to `b`.
    '''
    cdef size_t count = index_count

    copy_n(make_permutation_iterator(a._vector.begin(), index._vector.begin()),
           count, make_permutation_iterator(b._vector.begin(),
                                            index._vector.begin()))


cpdef sum_xy_vectors(DeviceVectorInt32 block_keys, DeviceVectorInt32 net_keys,
                     DeviceVectorInt32 p_x, DeviceVectorInt32 p_y,
                     DeviceVectorFloat32 e_x, DeviceVectorFloat32 e_x2,
                     DeviceVectorFloat32 e_y, DeviceVectorFloat32 e_y2,
                     DeviceVectorInt32 reduced_keys):
    cdef size_t count = net_keys.size
    cdef square[float] square_f
    cdef equal_to[int32_t] reduce_compare
    cdef reduce_plus4[float] reduce_plus4

    reduce_by_key(
        net_keys._vector.begin(),  # `keys_first`
        net_keys._vector.begin() + count,  # `keys_last`
        make_zip_iterator(  # `values_first`
            make_tuple4(
                make_permutation_iterator(p_x._vector.begin(),
                                          block_keys._vector.begin()),
                make_transform_iterator(
                    make_permutation_iterator(p_x._vector.begin(),
                                              block_keys._vector.begin()),
                    square_f),
                make_permutation_iterator(p_y._vector.begin(),
                                          block_keys._vector.begin()),
                make_transform_iterator(
                    make_permutation_iterator(p_y._vector.begin(),
                                              block_keys._vector.begin()),
                    square_f))),
        reduced_keys._vector.begin(),  # `keys_output`
        make_zip_iterator(make_tuple4(e_x._vector.begin(),
                                      e_x2._vector.begin(),
                                      e_y._vector.begin(),
                                      e_y2._vector.begin())),
        reduce_compare, reduce_plus4)


cpdef star_plus_2d(DeviceVectorFloat32 e_x, DeviceVectorFloat32 e_x2,
                   DeviceVectorFloat32 e_y, DeviceVectorFloat32 e_y2,
                   DeviceVectorFloat32 r_inv, float beta,
                   DeviceVectorFloat32 e_c):
    cdef size_t count = r_inv.size

    cdef c_star_plus_2d[float] *_star_plus = new c_star_plus_2d[float](beta)
    cdef unpack_quinary_args[c_star_plus_2d[float]] *_star_plus_2d = \
        new unpack_quinary_args[c_star_plus_2d[float]](deref(_star_plus))

    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple5(e_x._vector.begin(), e_x2._vector.begin(),
                            e_y._vector.begin(), e_y2._vector.begin(),
                            r_inv._vector.begin())),
            deref(_star_plus_2d)), count, e_c._vector.begin())
    del _star_plus
    del _star_plus_2d
    return <float>accumulate(e_c._vector.begin(), e_c._vector.begin() + count)


cpdef sum_permuted_float_by_key(DeviceVectorInt32 keys,
                                DeviceVectorFloat32 elements,
                                DeviceVectorInt32 index,
                                DeviceVectorInt32 reduced_keys,
                                DeviceVectorFloat32 reduced_values,
                                size_t key_count):
    cdef size_t count = <device_vector[int32_t].iterator>accumulate_by_key(
        keys._vector.begin(), keys._vector.begin() + key_count,
        make_permutation_iterator(elements._vector.begin(),
                                  index._vector.begin()),
        reduced_keys._vector.begin(), reduced_values._vector.begin()).first - reduced_keys._vector.begin()
    return count


def compute_block_group_keys(DeviceVectorUint32 block_slot_keys,
                             DeviceVectorUint32 block_slot_keys_prime,
                             DeviceVectorInt32 block_group_keys,
                             int32_t sentinel_key):
    cdef size_t count = block_slot_keys.size
    cdef block_group_key[int32_t] *group_key_func = \
        new block_group_key[int32_t](sentinel_key)

    transform2(block_slot_keys._vector.begin(), block_slot_keys._vector.begin()
               + count, block_slot_keys_prime._vector.begin(),
               block_group_keys._vector.begin(), deref(group_key_func))
    del group_key_func


cpdef equal_count_uint32(DeviceVectorUint32 a, DeviceVectorUint32 b):
    '''
    Return the number of index positions for which `a` and `b` are equal.

    Equivalent to:

        return (a == b).sum()
    '''
    cdef equal_to[uint32_t] _equal_to
    cdef unpack_binary_args[equal_to[uint32_t]] *unpacked_equal_to = \
        new unpack_binary_args[equal_to[uint32_t]](_equal_to)
    cdef identity[uint32_t] to_uint32
    cdef size_t count = a.size

    count = <size_t>accumulate(
        make_transform_iterator(
            make_transform_iterator(
                make_zip_iterator(make_tuple2(a._vector.begin(),
                                              b._vector.begin())),
                deref(unpacked_equal_to)), to_uint32),
        make_transform_iterator(
            make_transform_iterator(
                make_zip_iterator(make_tuple2(a._vector.begin() + count,
                                              b._vector.begin() + count)),
                deref(unpacked_equal_to)), to_uint32))
    del unpacked_equal_to
    return count


cpdef sequence_int32(DeviceVectorInt32 a):
    sequence(a._vector.begin(), a._vector.begin() + <size_t>a.size)


cpdef sort_netlist_keys(DeviceVectorInt32 keys1, DeviceVectorInt32 keys2):
    sort_by_key(keys1._vector.begin(), keys1._vector.begin() +
                <size_t>keys1.size, keys2._vector.begin())


cpdef permuted_nonmatch_inclusive_scan_int32(DeviceVectorInt32 elements,
                                             DeviceVectorInt32 index,
                                             DeviceVectorInt32 output,
                                             size_t index_count):
    cdef not_equal_to[int32_t] _not_equal_to
    cdef unpack_binary_args[not_equal_to[int32_t]] *unpacked_not_equal_to = \
        new unpack_binary_args[not_equal_to[int32_t]](_not_equal_to)
    cdef identity[int32_t] to_int32
    cdef size_t count = index_count - 1

    fill_n(output._vector.begin(), 1, 0)
    inclusive_scan(
        make_transform_iterator(
            make_transform_iterator(
                make_zip_iterator(
                    make_tuple2(
                        make_permutation_iterator(elements._vector.begin(),
                                                  index._vector.begin()),
                        make_permutation_iterator(elements._vector.begin(),
                                                  index._vector.begin() + 1))),
                deref(unpacked_not_equal_to)), to_int32),
        make_transform_iterator(
            make_transform_iterator(
                make_zip_iterator(
                    make_tuple2(
                        make_permutation_iterator(elements._vector.begin(),
                                                  index._vector.begin() +
                                                  count),
                        make_permutation_iterator(elements._vector.begin(),
                                                  index._vector.begin() + 1 +
                                                  count))),
                deref(unpacked_not_equal_to)), to_int32),
        output._vector.begin() + 1)
    del unpacked_not_equal_to


def assess_groups(float temperature, DeviceVectorInt32 group_block_keys,
                  DeviceVectorInt32 packed_block_group_keys,
                  DeviceVectorFloat32 group_delta_costs,
                  DeviceVectorInt32 output, size_t key_count):
    '''
    Given the specified annealing temperature and the delta cost for applying
    each group of associated-moves:

        - For each group of associated-moves, determine whether all moves in
          the group should be applied or rejected _(all or nothing)_.
        - Write the keys of all _blocks_ for which moves have been rejected to
          the `output` array.
        - Return the number of _blocks_ for which moves were rejected.

    Roughly equivalent to:

        N = group_delta_costs.size
        a = ((group_delta_costs <= 0) | (np.random.rand(N) <
                                         np.exp(-group_delta_costs /
                                                temperature)))
        rejected_block_keys = group_block_keys[~a[packed_block_group_keys]]
        return rejected_block_keys.size
    '''
    cdef size_t count = key_count
    cdef assess_group[float] *_assess_group = \
        new assess_group[float](temperature)
    cdef unpack_binary_args[assess_group[float]] *unpack_assess_group = \
        new unpack_binary_args[assess_group[float]](deref(_assess_group))
    cdef logical_not[uint8_t] _logical_not

    # a = ((group_delta_costs <= 0) | (rand() < np.exp(-group_delta_costs / temperature)))
    # rejected_block_keys = group_block_keys[~a[packed_block_group_keys]]
    count = <size_t>(copy_if_w_stencil(
        group_block_keys._vector.begin(), group_block_keys._vector.begin() +
        count,
        make_permutation_iterator(
            make_transform_iterator(
                make_zip_iterator(
                    make_tuple2(packed_block_group_keys._vector.begin(),
                                group_delta_costs._vector.begin())),
                deref(unpack_assess_group)),
            packed_block_group_keys._vector.begin()), output._vector.begin(),
        _logical_not) - output._vector.begin())
    del _assess_group
    del unpack_assess_group
    return count


cpdef sum_float_by_key(DeviceVectorInt32 keys, DeviceVectorFloat32 values,
                       DeviceVectorInt32 reduced_keys, DeviceVectorFloat32 reduced_values):
    cdef size_t count = (<device_vector[int32_t].iterator>accumulate_by_key(
        keys._vector.begin(), keys._vector.begin() + <size_t>keys.size,
        values._vector.begin(), reduced_keys._vector.begin(),
        reduced_values._vector.begin()).first - reduced_keys._vector.begin())
    return count


cpdef permuted_fill_float32(DeviceVectorFloat32 elements,
                          DeviceVectorInt32 index, float value):
    cdef size_t count = index.size

    fill_n(make_permutation_iterator(elements._vector.begin(),
                                     index._vector.begin()),
           count, value)


cpdef permuted_fill_int32(DeviceVectorInt32 elements,
                          DeviceVectorInt32 index, int32_t value):
    cdef size_t count = index.size

    fill_n(make_permutation_iterator(elements._vector.begin(),
                                     index._vector.begin()),
           count, value)


cpdef permuted_fill_int8(DeviceVectorInt8 elements, DeviceVectorInt32 index,
                         int8_t value):
    cdef size_t count = index.size

    fill_n(make_permutation_iterator(elements._vector.begin(),
                                     index._vector.begin()),
           count, value)


cpdef look_up_delay(DeviceVectorInt32 i_index, DeviceVectorInt32 j_index,
                    DeviceVectorInt32 p_x, DeviceVectorInt32 p_y,
                    DeviceVectorFloat32 delays,
                    int32_t nrows, int32_t ncols,
                    DeviceVectorInt8 delay_type,
                    DeviceVectorFloat32 delays_ij):
    cdef size_t count = i_index.size
    cdef absolute[int32_t] abs_func
    cdef minus[int32_t] minus_func
    cdef unpack_binary_args[minus[int32_t]] *unpacked_minus = \
        new unpack_binary_args[minus[int32_t]](minus_func)
    cdef delay[device_vector[float].iterator] *delay_f = \
        new delay[device_vector[float].iterator](delays._vector.begin(),
                                                 nrows, ncols)
    cdef unpack_ternary_args[delay[device_vector[float].iterator]] \
        *unpacked_delay = new \
        unpack_ternary_args[delay[device_vector[float].iterator]]\
        (deref(delay_f))

    copy_n(
        make_transform_iterator(
            make_zip_iterator(
                make_tuple3(
                    delay_type._vector.begin(),
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_x._vector.begin(),
                                        i_index._vector.begin()),
                                    make_permutation_iterator(
                                        p_y._vector.begin(),
                                        i_index._vector.begin()))),
                            deref(unpacked_minus)), abs_func),
                    make_transform_iterator(
                        make_transform_iterator(
                            make_zip_iterator(
                                make_tuple2(
                                    make_permutation_iterator(
                                        p_x._vector.begin(),
                                        j_index._vector.begin()),
                                    make_permutation_iterator(
                                        p_y._vector.begin(),
                                        j_index._vector.begin()))),
                            deref(unpacked_minus)), abs_func))),
            deref(unpacked_delay)), count, delays_ij._vector.begin())
    del unpacked_minus
    del delay_f
    del unpacked_delay
