# Installation #

Although `camip` dependencies are listed under `install_requires` and
`requirements.txt`, some packages must be installed beforehand due to
setup-time dependencies.  To install the setup-time dependencies, execute the
following commands in a terminal:

    pip install "numpy>=1.9.0" "jinja2>=2.7.3" "Cython>=0.21" "pandas>=0.14.1" "paver>=1.2.3" "numexpr>=2.0.0"

After installing setup dependencies, `camip` can be installed from the Python
package index using `pip`, as follows:

    pip install camip


# Usage #

After installing `camip`, a [VPR v5 format `.net` netlist file][1] can be
placed using the `camip.bin.place_vpr_net` script.  The usage information for
the `camip.bin.place_vpr_net` script is shown below:


    python -m camip.bin.place_vpr_net [-h] [-o OUTPUT_PATH | -d OUTPUT_DIR]
                                      [-i INNER_NUM] [-C] [-I IO_CAPACITY]
                                      [-s SEED]
                                      vpr_net_file vpr_arch_file

    Run Concurrent Associated Moves Iterative Placement (CAMIP) on VPR v5
    netlist (`.net`) file.

    positional arguments:
      vpr_net_file
      vpr_arch_file

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT_PATH, --output_path OUTPUT_PATH
      -d OUTPUT_DIR, --output_dir OUTPUT_DIR
      -i INNER_NUM, --inner-num INNER_NUM
      -I IO_CAPACITY, --io-capacity IO_CAPACITY
      -s SEED, --seed SEED

## Input ##

Below is an example netlist in [VPR v5 `.net` format][1].  Please see [here][1]
for details on the file format.

    # `test.net` - Example netlist with 14 blocks
    .global clk

    .input block_0
    pinlist: clk

    .input block_1
    pinlist:    net_1

    .input block_2
    pinlist:    net_2

    .clb block_3
    pinlist:    net_1  open  open  open   net_3  open
    subblock: block_3 0 open open open  4  open

    .clb block_4
    pinlist:    net_1   net_2   net_8  open   net_4  open
    subblock: block_4 0 1 2 open  4  open

    .clb block_5
    pinlist:    net_1   net_2  open  open   net_5  open
    subblock: block_5 0 1 open open  4  open

    .clb block_6
    pinlist:    net_3   net_4  open  open   net_6  clk
    subblock: block_6 0 1 open open  4 5

    .clb block_7
    pinlist:    net_4  open  open  open   net_7  open
    subblock: block_7 0 open open open  4  open

    .clb block_8
    pinlist:    net_5  open  open  open   net_8  clk
    subblock: block_8 0 open open open  4 5

    .clb block_9
    pinlist:    net_6   net_1  open  open   net_9  open
    subblock: block_9 0 1 open open  4  open

    .clb block_10
    pinlist:    net_7   net_8  open  open   net_10  clk
    subblock: block_10 0 1 open open  4 5

    .output block_11
    pinlist:    net_9

    .output block_12
    pinlist:    net_7

    .output block_13
    pinlist:    net_10

## Output ##

The `camip.bin.place_vpr_net` script outputs two files:

 1. A `.out` file in [VPR v5 placement format][1].
 2. A `.h5` file in [HDF5][2] format.

The `.out` file contains an ASCII representation of the block positions in [VPR
v5 placement format][1].  This file can be routed using VPR v5 using the
`-route_only` option.  See the [VPR project documentation][2] for more
information.

The `.h5` file is not required for routing, but contains the final block
positions and additional statistics gathered during placement.  The file is in
HDF5 format and can be browsed using tools such as [`HDFView`][3] or
[`ViTables`][4].  For in-depth analysis, the data tables in the HDF5 file can
be opened, for example, using the Python `pandas` package.  See below for
details.

Note that example output files for the `.net` file listed above can be found
[here (`.out`)][.out] and [here (`.h5`)][.h5].


### VPR `.out` placement output ###

The following is an example placement in [VPR v5 `.out` format][1].  Please see
[here][1] for details on the file format.

    Netlist file: test.net Architecture file: 4lut_sanitized.arch
    Array size: 3 x 3 logic blocks

    #block name     x   y   subblk
    #----------     --  --  ------
    block_0         0   3   0
    block_1         4   2   0
    block_2         4   2   1
    block_3         1   1   0
    block_4         2   2   0
    block_5         3   2   0
    block_6         2   1   0
    block_7         1   2   0
    block_8         2   3   0
    block_9         3   1   0
    block_10        1   3   0
    block_11        4   1   0
    block_12        0   3   1
    block_13        1   4   0


### HDF output ###

 - `/params`: Parameters used for placement, including random seed, starting
   temperature, and starting radius limit.
 - `/block_positions`: A table containing the final position of each block
   (i.e., the placement).  Each row corresponds to a block and contains the
   following fields:
   * `index` (`string`): Block name from the `.net file.
   * `x` (`uint32`): Block `x` position.
   * `y` (`uint32`): Block `y` position.
   * `z` (`uint32`): Block `z` position (i.e., subblock).
   * `type` (`uint32`):
     - **0**: Logic block.
     - **1**: Input block.
     - **2**: Output block.
 - `/place_stats`: A table containing the final position of each block
   (i.e., the placement).  Each row corresponds to an outer loop iteration of
   the simulated anneal and contains the following fields:
   * `index` (`int64`): Iteration index.
   * `start` (`float64`): Iteration start time (seconds since epoch).
   * `end` (`float64`): Iteration end time (seconds since epoch).
   * `temperature` (`float64`): Iteration temperature.
   * `cost` (`float64`): Cost after iteration.
   * `success_ratio` (`float64`): Fraction of moves accepted in the iteration.
   * `radius_limit` (`float64`): Iteration radius limit (i.e., max move distance
     in either dimension).
   * `evaluated_move_count` (`int64`): Number moves evaluated during iteration.

Note that the `start` and `end` times correspond to the time spent include all
calculations that contribute to placement, but do not include time to record
the placement statistics for each iteration.

**The total placement time (not including loading of the netlist and initial
population of data structures) can be computed by summing the difference of the
end and start time of each iteration.**


[1]: http://www.eecg.toronto.edu/~vaughn/challenge/netlist.html
[2]: http://www.hdfgroup.org/HDF5/
[3]: http://www.hdfgroup.org/products/java/hdfview/
[4]: http://vitables.org/
[.out]: https://github.com/cfobel/camip/blob/master/camip/tests/fixtures/placed-13blocks.out
[.h5]: https://github.com/cfobel/camip/blob/master/camip/tests/fixtures/placed-13blocks.h5
