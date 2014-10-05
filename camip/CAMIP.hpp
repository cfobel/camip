#ifndef ___CAMIP__HPP___
#define ___CAMIP__HPP___

#include <cmath>
#include <cstdlib>  //  RAND_MAX
#include <thrust/pair.h>


struct evaluate_move {
  typedef float result_type;

  float beta;

  evaluate_move(float beta=1.59) : beta(beta) {}

  result_type operator() (float e, float e2, int32_t p, int32_t p_prime,
                          float r_inv) {
    float new_e = e - p + p_prime;
    float new_e_2 = e2 - p * p + p_prime * p_prime;
    float result = sqrt(new_e_2 - new_e * new_e * r_inv + 1);
    return beta * result;
  }
};


template <typename VPRMovePattern>
struct slot_move {
  typedef int32_t result_type;

  VPRMovePattern move_pattern;

  slot_move(VPRMovePattern move_pattern) : move_pattern(move_pattern) {}

  result_type operator() (int32_t slot) {
    return move_pattern(slot);
  }
};


template <typename T>
class IOSegmentStarts {
public:
  int bottom;
  int right;
  int top;
  int left;

  IOSegmentStarts() {}
  IOSegmentStarts(T io_count)
    : bottom(0), right(io_count.first), top(right + io_count.second),
      left(top + io_count.first) {}
};


template <typename Pair>
class VPRIOSlotKeyTo2dPosition {
public:
  typedef Pair result_type;
  Pair extent;
  int io_capacity;
  IOSegmentStarts<Pair> segment_start;
  int size;

  VPRIOSlotKeyTo2dPosition() {}

  VPRIOSlotKeyTo2dPosition(Pair extent, int io_capacity)
    : extent(extent), io_capacity(io_capacity) {
      Pair io_count = thrust::make_pair(extent.first * io_capacity,
                                        extent.second * io_capacity);
      segment_start = IOSegmentStarts<Pair>(io_count);
      size = 2 * (io_count.first + io_count.second);
  }

  result_type operator() (int k) const {
    if (k < segment_start.right) {
        /* Slot `k` maps to position along the 'bottom' of the grid. */
        return thrust::make_pair(0, 1 + k / io_capacity);
    } else if (k < segment_start.top) {
        /* Slot `k` maps to position along the ['right'] side of the grid. */
        return thrust::make_pair(1 + (k - segment_start.right) / io_capacity,
                                 extent.first + 1);
    } else if (k < segment_start.left) {
        /* Slot `k` maps to position along the top of the grid. */
        return thrust::make_pair(extent.second + 1,
                                 extent.first - (k - segment_start.top) /
                                 io_capacity);
    } else {
        /* Assume slot `k` maps to position along the left of the grid. */
        return thrust::make_pair(extent.second - (k - segment_start.left) /
                                 io_capacity, 0);
    }
  }
};


template <typename Pair>
class SlotKeyTo2dPosition {
public:
  typedef Pair result_type;
  int row_extent;
  Pair offset;

  SlotKeyTo2dPosition() {}
  SlotKeyTo2dPosition(int row_extent, Pair offset)
    : row_extent(row_extent), offset(offset) {}

  SlotKeyTo2dPosition(int row_extent)
    : row_extent(row_extent), offset(thrust::make_pair(0, 0)) {}

  Pair operator() (int k) const {
      return thrust::make_pair(k / row_extent + offset.first, k % row_extent
                                + offset.second);
  }
};


class BlockTypeCount {
public:
  int io;
  int logic;

  BlockTypeCount(int io=0, int logic=0) : io(io), logic(logic) {}

  BlockTypeCount operator+ (const BlockTypeCount &other) const {
    return BlockTypeCount(io + other.io, logic + other.logic);
  }

  BlockTypeCount operator- (const BlockTypeCount &other) const {
    return BlockTypeCount(io - other.io, logic - other.logic);
  }
};


template <typename Pair>
struct VPRAutoSlotKeyTo2dPosition {
  typedef Pair result_type;

  int io_count;
  int logic_count;
  int io_capacity;
  Pair extent;
  VPRIOSlotKeyTo2dPosition<Pair> io_s2p;
  SlotKeyTo2dPosition<Pair> logic_s2p;
  BlockTypeCount slot_count;

  VPRAutoSlotKeyTo2dPosition() {}
  VPRAutoSlotKeyTo2dPosition(int io_count, int logic_count, int io_capacity=2)
    : io_count(io_count), logic_count(logic_count), io_capacity(io_capacity) {
      extent = Pair();
      extent.first = static_cast<int>(ceil(sqrt(logic_count)));
      extent.second = extent.first;

      if (io_count > 2 * io_capacity * (extent.first + extent.second)) {
        /* The size determined based on the number of logic blocks does not
         * provide enough spots for the inputs/outputs along the perimeter.
         * Increase extents of the grid to fit IO. */
        extent.first = static_cast<int>(ceil(sqrt(io_count + logic_count)));
        extent.second = extent.first;
      }

      Pair io_extent;

      if (io_count > 0) { io_extent = extent; }

      io_s2p = VPRIOSlotKeyTo2dPosition<Pair>(io_extent, io_capacity);

      /* Logic grid starts at `(1, 1)`. */
      logic_s2p = SlotKeyTo2dPosition<Pair>(extent.first,
                                            thrust::make_pair(1, 1));
      slot_count = BlockTypeCount(io_s2p.size, extent.first * extent.second);
  }

  size_t size() const { return slot_count.io + slot_count.logic; }

  Pair position0(Pair position) const {
    return thrust::make_pair(position.first - logic_s2p.offset.first,
                             position.second - logic_s2p.offset.second);
  }

  bool in_bounds(Pair position) const {
    Pair p = position0(position);
    return !(p.first < 0 || p.first >= extent.second || p.second < 0 ||
             p.second >= extent.first);
  }

  Pair get0(int k) const { return position0((*this)(k)); }

  Pair operator() (int k) const {
    if (k < slot_count.io) {
      return io_s2p(k);
    } else {
      return logic_s2p(k - slot_count.io);
    }
  }
};


class MovePattern {
public:
  typedef int result_type;
  int magnitude;
  int double_magnitude;
  int shift;

  MovePattern() {}
  MovePattern(int magnitude, int shift=0)
    : magnitude(magnitude), double_magnitude(2 * magnitude), shift(shift) {}

  int operator() (int i) const {
    if (magnitude == 0) { return 0; }
    int index = (i + 2 * double_magnitude - ((shift + magnitude + 1) %
                 double_magnitude));
    if ((index % double_magnitude) < magnitude) {
      return magnitude;
    } else {
      return -magnitude;
    }
  }
};


template <typename Pair>
class MovePattern2d {
public:
  typedef int result_type;
  MovePattern row;
  MovePattern column;
  Pair extent;
  int size;

  MovePattern2d() {}
  MovePattern2d(Pair magnitude, Pair shift, Pair extent)
    : row(MovePattern(magnitude.first, shift.first)),
      column(MovePattern(magnitude.second, shift.second)), extent(extent),
      size(extent.first * extent.second) {}

  int column_i(int i) const {
    return ((i / extent.first) + extent.second * (i % extent.first));
  }

  int get_column(int i) const { return column(column_i(i)); }

  int get_row(int i) const { return row(i); }

  int operator() (int i) const { return Pair(get_row(i), get_column(i)); }
};


class MovePatternInBounds {
public:
  typedef int result_type;
  int extent;
  MovePattern pattern;

  MovePatternInBounds() {}
  MovePatternInBounds(int extent, int magnitude, int shift=0)
    : extent(extent), pattern(MovePattern(magnitude, shift)) {}

  int operator() (int i) const {
    /* We still need to use the offset-based position for computing the target
     * position. */
    int move = pattern(i);
    int target = i + move;
    if (target < 0 || target >= extent) {
      /* If the displacement targets a location that is outside the boundaries,
       * set displacement to zero. */
      return 0;
    }
    return move;
  }

  size_t size() const { return extent; }
};


template <typename Pair>
struct MovePatternInBounds2d {
  VPRAutoSlotKeyTo2dPosition<Pair> s2p;
  MovePattern2d<Pair> pattern;

  MovePatternInBounds2d() {}
  MovePatternInBounds2d(Pair magnitude, Pair shift,
                        VPRAutoSlotKeyTo2dPosition<Pair>
                        slot_key_to_position)
    : s2p(slot_key_to_position),
      pattern(MovePattern2d<Pair>(magnitude, shift, s2p.extent)) {}

  int32_t operator() (int32_t k) const {
    /* Get zero-based position, since displacement patterns are indexed
     * starting at zero. */
    Pair position0 = s2p.get0(k);

    /* We still need to use the offset-based position for computing the target
     * position. */
    Pair position = s2p(k);
    Pair move(pattern.column(position0.first), pattern.row(position0.second));
    Pair target(position.first + move.first, position.second + move.second);

    if (!s2p.in_bounds(target)) {
      /* If the displacement targets a location that is outside the boundaries,
       * set displacement to zero. */
      return 0;
    }
    return move.first * s2p.extent.first + move.second;
  }

  size_t size() const { return pattern.size; }
};


class PatternParams {
public:
  int32_t magnitude;
  int32_t shift;
};


template <typename Pair>
class PatternParams2D {
public:
  Pair magnitude;
  Pair shift;

  PatternParams2D() {}
};


template <typename Pair>
struct VPRMovePattern {
  typedef int32_t result_type;

  VPRAutoSlotKeyTo2dPosition<Pair> s2p;
  MovePatternInBounds io_pattern;
  MovePatternInBounds2d<Pair> logic_pattern;

  VPRMovePattern() {}
  VPRMovePattern(int io_magnitude, int io_shift, Pair logic_magnitude,
                 Pair logic_shift,
                 VPRAutoSlotKeyTo2dPosition<Pair> slot_key_to_position)
    : s2p(slot_key_to_position),
      io_pattern(MovePatternInBounds(io_slot_count(), io_magnitude, io_shift)),
      logic_pattern(MovePatternInBounds2d<Pair>(logic_magnitude, logic_shift,
                                                s2p)) {}

  int32_t operator() (int32_t k) {
    if (k < io_slot_count()) {
      return io_pattern(k);
    } else {
      return logic_pattern(k);
    }
  }

  size_t io_slot_count() const { return s2p.slot_count.io; }
  size_t logic_slot_count() const { return s2p.slot_count.logic; }

  size_t total_slot_count() const {
    return io_slot_count() + logic_slot_count();
  }
};


template <typename T>
struct block_group_key {
  typedef T result_type;
  T sentinel_key;

  block_group_key(T sentinel_key) : sentinel_key(sentinel_key) {}

  result_type operator() (T k1, T k2) {
    if (k1 != k2) {
      return (k1 < k2) ? k1 : k2;
    } else {
      return sentinel_key;
    }
  }
};


template <typename T>
struct star_plus_2d {
  typedef T result_type;

  float beta;

  star_plus_2d(float beta=1.59) : beta(beta) {}

  result_type operator() (T e_x, T e_x2, T e_y, T e_y2, T r_inv) {
      return beta * (sqrt(e_x2 - e_x * e_x * r_inv + 1) +
                     sqrt(e_y2 - e_y * e_y * r_inv + 1));
  }
};


template <typename RealT>
struct assess_group {
  /* # `assess_move_pair_functor` # */
  /* Assess whether a move-pair should be applied, based on the current
    * temperature setting.  The assessment criteria corresponds to the
    * annealing acceptance rules from VPR. */
  typedef bool result_type;

  RealT temperature_;

  assess_group(RealT temperature) : temperature_(temperature) {}

  bool assess(size_t seed, RealT delta) {
    /* ## `assess` ## */
    /* Based on a `seed` and a scalar delta-cost, decide whether or not the
      * corresponding swap should be accepted. */
    if (delta <= 0) {
        return true;
    }

    if (temperature_ == 0.) {
        return false;
    }

    /* - Generate a random floating-point value between `0` and `1`. */
    RealT fnum = SimpleRNG<uint32_t, RealT>()(seed);
    RealT prob_fac = exp(-delta / temperature_);
    return (prob_fac > fnum);
  }

  template <typename Key, typename Delta>
  result_type operator() (Key group_key, Delta delta) {
    /* Apply simple hash function to compute a random-number generator seed
     * based on group key and group delta cost. */
    size_t seed = static_cast<size_t>(delta * static_cast<RealT>(RAND_MAX));
    seed ^= static_cast<size_t>(group_key);

    return assess(seed, delta);
  }
};


#endif  // #ifndef ___CAMIP__HPP___
