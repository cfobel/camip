# coding: utf-8
import hashlib
import sys

from path_helpers import path
from table_layouts import (get_PLACEMENT_TABLE_LAYOUT,
                           get_PLACEMENT_STATS_TABLE_LAYOUT,
                           get_PLACEMENT_STATS_DATAFRAME_LAYOUT)
from camip import CAMIP, VPRSchedule
from camip.timing import CAMIPTiming
from camip.device.CAMIP import extract_positions
import numpy as np
import tables as ts
import pandas as pd
from cyplace_experiments.data.connections_table import ConnectionsTable


def place(net_file_namebase, seed, io_capacity=3, inner_num=1.,
          include_clock=False, timing=False, critical_path_only=False,
          wire_length_factor=0.5, criticality_exp=10.):
    connections_table = ConnectionsTable.from_net_list_name(net_file_namebase)
    if timing or critical_path_only:
        if timing:
            critical_path_only = False
        placer = CAMIPTiming(connections_table, io_capacity=io_capacity,
                             include_clock=include_clock,
                             timing_cost_disabled=critical_path_only,
                             wire_length_factor=wire_length_factor,
                             criticality_exp=criticality_exp)
    else:
        placer = CAMIP(connections_table, include_clock=include_clock,
                       io_capacity=io_capacity)
    placer.shuffle_placement()
    print placer.evaluate_placement()
    schedule = VPRSchedule(placer.s2p, inner_num, placer.block_count, placer)
    print 'starting temperature: %.2f' % schedule.anneal_schedule.temperature

    states = schedule.run(placer)

    # Convert list of state dictionaries into a `pandas.DataFrame`.
    stats_layout = pd.DataFrame(
        np.array([], dtype=get_PLACEMENT_STATS_DATAFRAME_LAYOUT()))
    place_stats = stats_layout.append(states)

    return placer, place_stats


def save_placement(net_file_namebase, block_positions, place_stats,
                   output_path=None, output_dir=None, inner_num=1., seed=0):
    '''
    Perform placement and write result to HDF file with the following
    structure:

        <net-file_namebase _(e.g., `ex5p`, `clma`, etc.)_> (Group)
            \--> `placements` (Table)

    The intention here is to structure the results such that they can be merged
    together with the results from other placements.
    '''
    # Use a hash of the block-positions to name the HDF file.
    block_positions_sha1 = hashlib.sha1(block_positions
                                        .astype('uint32').data).hexdigest()

    filters = ts.Filters(complib='blosc', complevel=6)
    if output_path is not None:
        context = dict(net_file_namebase=net_file_namebase, seed=seed,
                       block_positions_sha1=block_positions_sha1)
        output_path = str(output_path) % context
    else:
        output_file_name = 'placed-%s-s%d-%s.h5' % (net_file_namebase, seed,
                                                    block_positions_sha1)
        output_path = output_file_name
    if output_dir is not None:
        output_path = str(output_dir.joinpath(output_path))

    parent_dir = path(output_path).parent
    if parent_dir and not parent_dir.isdir():
        parent_dir.makedirs_p()
    print 'writing output to: %s' % output_path

    h5f = ts.open_file(output_path, mode='w', filters=filters)

    net_file_results = h5f.create_group(h5f.root, net_file_namebase,
                                        title='Placement results for %s CAMIP '
                                        'with `inner_num`=%s, wire-length '
                                        'driven' % (net_file_namebase,
                                                    inner_num))

    TABLE_LAYOUT = get_PLACEMENT_TABLE_LAYOUT(len(block_positions)),
    placements = h5f.create_table(
        net_file_results, 'placements',
        get_PLACEMENT_TABLE_LAYOUT(len(block_positions)),
        title='Placements for %s VPR' % net_file_namebase)
    placements.set_attr('net_file_namebase', net_file_namebase)

    placements.cols.block_positions_sha1.create_index()
    row = placements.row
    row['net_list_name'] = net_file_namebase
    row['block_positions'] = block_positions
    row['block_positions_sha1'] = block_positions_sha1
    row['seed'] = seed
    # Convert start-date-time to UTC unix timestamp
    row['start'] = place_stats['start'].iat[0]
    row['end'] = place_stats['end'].iat[-1]
    row['inner_num'] = inner_num
    row.append()
    placements.flush()

    stats_group = h5f.create_group(net_file_results, 'placement_stats',
                                  title='Placement statistics for each '
                                  'outer-loop iteration of a anneal for %s' %
                                  net_file_namebase)

    # Prefix `block_positions_sha1` with `P_` to ensure the table-name is
    # compatible with Python natural-naming.  This is necessary since SHA1
    # hashes may start with a number, in which case the name would not be a
    # valid Python attribute name.
    placement_stats = h5f.create_table(stats_group, 'P_' + block_positions_sha1,
                                      get_PLACEMENT_STATS_TABLE_LAYOUT(),
                                      title='Placement statistics for each '
                                      'outer-loop iteration of a VPR anneal '
                                      'for %s with which produced the '
                                      'block-positions with SHA1 hash `%s`'
                                      % (net_file_namebase,
                                         block_positions_sha1))
    placement_stats.set_attr('net_file_namebase', net_file_namebase)
    placement_stats.set_attr('block_positions_sha1', block_positions_sha1)

    for i, stats in place_stats.iterrows():
        stats_row = placement_stats.row
        for field in ('start', 'end', 'temperature', 'cost', 'success_ratio',
                      'radius_limit', 'total_iteration_count'):
            stats_row[field] = stats[field] if not np.isnan(stats[field]) else -1
        stats_row.append()
    placement_stats.flush()

    h5f.close()


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
    parser.add_argument('-I', '--io-capacity', type=int, default=3)
    parser.add_argument('-s', '--seed', default=np.random.randint(100000),
                        type=int)
    parser.add_argument(dest='net_file_namebase')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print args

    np.random.seed(args.seed)
    placer, place_stats = place(args.net_file_namebase, args.seed,
                                args.io_capacity, args.inner_num,
                                args.include_clock, args.timing,
                                args.critical_path, args.wire_length_factor,
                                args.criticality_exp)

    extract_positions(placer.block_slot_keys, placer.p_x, placer.p_y,
                      placer.s2p)
    p_z = np.zeros(placer.p_x.size, dtype=np.uint32)
    p_z[:placer.s2p.io_count] = (placer.block_slot_keys[:placer.s2p.io_count] %
                                 placer.io_capacity)
    block_positions = np.array([placer.p_x[:], placer.p_y[:], p_z],
                               dtype='uint32').T
    if args.output_path is None and args.timing:
        args.output_path = ('placed-%(net_file_namebase)s-s%(seed)s-%(block_positions_sha1)s' +
                            '-e%.2f-w%.2f-I%d.h5' % (args.criticality_exp,
                                                     args.wire_length_factor,
                                                     args.io_capacity))
    save_placement(args.net_file_namebase, block_positions, place_stats,
                   output_path=args.output_path, output_dir=args.output_dir,
                   inner_num=args.inner_num, seed=args.seed)
    #if args.draw_enabled:
        #raw_input()
