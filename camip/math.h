#ifndef ___UTILITY__MATH__HPP___
#define ___UTILITY__MATH__HPP___

inline double get_std_dev(int n, double sum_x_squared, double av_x) {
  /* # `get_std_dev` #
   *
   * From [VRP][1]:
   *
   * > Returns the standard deviation of data set x.  There are n sample
   * > points, sum_x_squared is the summation over n of x^2 and av_x is the
   * > average x. All operations are done in double precision, since round off
   * > error can be a problem in the initial temp. std_dev calculation for big
   * > circuits.
   *
   * [1]: https://code.google.com/p/vtr-verilog-to-routing/ */

  double std_dev;

  if(n <= 1) {
    std_dev = 0.;
  } else {
    std_dev = (sum_x_squared - n * av_x * av_x) / (double)(n - 1);
  }

  if(std_dev > 0.) {
    /* Very small variances sometimes round negative */
    std_dev = sqrt(std_dev);
  } else {
    std_dev = 0.;
  }

  return (std_dev);
}

#endif  // #ifndef ___UTILITY__MATH__HPP___
