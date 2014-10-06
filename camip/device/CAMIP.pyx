#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t, int32_t, uint8_t
from libc.math cimport fmin
import numpy as np
cimport numpy as np

from cythrust.device_vector cimport (DeviceVectorInt32, DeviceVectorUint32,
                                     DeviceVectorFloat32)
from cythrust.thrust.iterator.repeated_range_iterator cimport repeated_range
from cythrust.thrust.iterator.counting_iterator cimport counting_iterator
from cythrust.thrust.sort cimport sort_by_key, sort
from cythrust.thrust.scan cimport exclusive_scan, inclusive_scan
from cythrust.thrust.reduce cimport accumulate, accumulate_by_key, reduce_by_key
from cythrust.thrust.iterator.transform_iterator cimport make_transform_iterator
from cythrust.thrust.copy cimport copy_n, copy_if_w_stencil
from cythrust.thrust.sequence cimport sequence
from cythrust.thrust.transform cimport transform, transform2
from cythrust.thrust.iterator.permutation_iterator cimport make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.thrust.tuple cimport make_tuple5, make_tuple4, make_tuple2
from cythrust.thrust.functional cimport (unpack_binary_args, square, equal_to,
                                         not_equal_to, unpack_quinary_args,
                                         plus, minus, reduce_plus4, identity,
                                         logical_not)
from camip.CAMIP cimport VPRAutoSlotKeyTo2dPosition, evaluate_move



cpdef evaluate_moves(DeviceVectorInt32 row, DeviceVectorInt32 col,
                     DeviceVectorInt32 p_x, DeviceVectorInt32 p_x_prime,
                     DeviceVectorFloat32 e_x, DeviceVectorFloat32 e_x_2,
                     DeviceVectorInt32 p_y, DeviceVectorInt32 p_y_prime,
                     DeviceVectorFloat32 e_y, DeviceVectorFloat32 e_y_2,
                     DeviceVectorFloat32 r_inv, float beta,
                     DeviceVectorInt32 reduced_keys,
                     DeviceVectorFloat32 reduced_values):
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


def minus_float(DeviceVectorFloat32 n_c, DeviceVectorFloat32 n_c_prime,
                DeviceVectorFloat32 delta_n):
    cdef size_t count = delta_n.size
    cdef minus[float] minus_func

    transform2(n_c._vector.begin(), n_c._vector.begin() + count,
               n_c_prime._vector.begin(), delta_n._vector.begin(), minus_func)
