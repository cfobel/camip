# coding: utf-8
import sys

from path_helpers import path
import numpy as np
from camip.place import (block_positions_df_to_vpr, place_from_vpr_net,
                         placer_to_block_positions_df, save_placement)


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Run CAMIP place based on net-file'
                            'namebase.')
    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument('-o', '--output_path', default=None, type=path)
    mutex_group.add_argument('-d', '--output_dir', default=None, type=path)
    parser.add_argument('-e', '--criticality-exp', type=float, default=22.5)
    parser.add_argument('-i', '--inner-num', type=float, default=1.)
    parser.add_argument('-C', '--include-clock', action='store_true',
                        help='Include clock in wire-length optimization.')
    parser.add_argument('-c', '--critical-path', action='store_true',
                        help='Enable critical path calculation.')
    parser.add_argument('-t', '--timing', action='store_true',
                        help='Optimize using path-based delays.')
    parser.add_argument('-w', '--wire-length-factor', type=float, default=0.5,
                        help='When timing is enabled, fraction of emphasis to '
                        'place on wire-length _(vs. timing)_.')
    parser.add_argument('-I', '--io-capacity', type=int, default=2)
    parser.add_argument('-s', '--seed', default=np.random.randint(100000),
                        type=int)
    parser.add_argument(dest='vpr_net_file', type=path)
    parser.add_argument(dest='vpr_arch_file', type=path)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print args

    schedule, placer, params, place_stats = place_from_vpr_net(
        args.vpr_net_file, args.seed,
        io_capacity=args.io_capacity, inner_num=args.inner_num,
        include_clock=args.include_clock, timing=args.timing,
        critical_path_only=args.critical_path,
        wire_length_factor=args.wire_length_factor,
        criticality_exp=args.criticality_exp)

    block_positions = placer_to_block_positions_df(placer)

    if args.output_path is None and args.timing:
        args.output_path = ('placed-%(net_file_namebase)s-s%(seed)s-%(block_positions_sha1)s' +
                            '-e%.2f-w%.2f-I%d.h5' % (args.criticality_exp,
                                                     args.wire_length_factor,
                                                     args.io_capacity))
    hdf_output_path = save_placement(args.vpr_net_file.namebase,
                                     block_positions, params, place_stats,
                                     output_path=args.output_path,
                                     output_dir=args.output_dir)

    vpr_placement_path = ('%s.out' % hdf_output_path.parent
                          .joinpath(hdf_output_path.namebase))
    print 'writing VPR placement output to:', vpr_placement_path
    block_positions_df_to_vpr(block_positions, placer.s2p.extent,
                              args.vpr_net_file, args.vpr_arch_file,
                              vpr_placement_path)
