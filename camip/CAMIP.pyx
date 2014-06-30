#cython: embedsignature=True, boundscheck=False
from libc.stdint cimport uint32_t, int32_t
cimport cython

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)


cdef inline evaluate_move(int i, int j, int32_t[:] p, int32_t[:] p_prime,
                          float[:] e, float[:] e_2, float[:] r_inv):
    cdef float new_e = (e[j] - p[i] + p_prime[i])
    cdef float result = sqrt(new_e * new_e - (e_2[j] - p[i] * p[i] + p_prime[i]
                                              * p_prime[i]) * r_inv[j] + 1)
    return result


cpdef evaluate_moves(float[:] output, int32_t[:] row, int32_t[:] col,
                     int32_t[:] p_x, int32_t[:] p_x_prime,
                     float[:] e_x, float[:] e_x_2,
                     int32_t[:] p_y, int32_t[:] p_y_prime,
                     float[:] e_y, float[:] e_y_2,
                     float[:] r_inv, float beta):
    cdef int count = len(output)
    cdef int k
    cdef int i
    cdef int j

    for k in xrange(count):
        i = row[k]
        j = col[k]
        output[k] = beta * (evaluate_move(i, j, p_x, p_x_prime, e_x, e_x_2,
                                          r_inv) +
                            evaluate_move(i, j, p_y, p_y_prime, e_y, e_y_2,
                                          r_inv))
