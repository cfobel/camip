#ifndef ___CAMIP_TIMING__H___
#define ___CAMIP_TIMING__H___


template <typename DelayIterator>
struct delay {
  typedef float result_type;
  DelayIterator delays;
  int32_t nrows;
  int32_t ncols;

  delay(DelayIterator delays, int32_t nrows, int32_t ncols)
    : delays(delays), nrows(nrows), ncols(ncols) {}

  template <typename T1, typename T2, typename T3>
  result_type operator() (T1 delay_type, T2 delta_x, T3 delta_y) {
    size_t stride = 0;
    size_t offset = 0;

    if (delay_type == 0) {
      /* Delay is logic-to-logic. */
      stride = ncols;
      offset = 0;
    } else if (delay_type == 20) {
      /* Delay is logic-to-io. */
      stride = ncols + 1;
      offset = nrows * ncols;
    } else if (delay_type == 1) {
      /* Delay is io-to-io. */
      stride = ncols + 1;
      offset = nrows * ncols + (nrows + 1) * (ncols + 1);
    } else if (delay_type == 21) {
      /* Delay is io-to-io. */
      stride = ncols + 2;
      offset = nrows * ncols + 2 * (nrows + 1) * (ncols + 1);
    }
    return *(delays + offset + stride * delta_x + delta_y);
  }
};


struct arrival_delay {
  typedef float result_type;

  template <typename T1, typename T2, typename T3>
  result_type operator() (T1 j_is_sync, T2 delay_ij, T3 t_a_j) {
    if (j_is_sync) { return delay_ij; }
    else if (t_a_j > 0) { return t_a_j + delay_ij; }
    else { return 0; }
  }
};

#endif  // #ifndef ___CAMIP_TIMING__H___
