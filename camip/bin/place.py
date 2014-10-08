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


def place(net_file_namebase, seed, inner_num=1., timing=False,
          draw_enabled=False):
    if timing:
        placer = CAMIPTiming(net_file_namebase)
    else:
        placer = CAMIP(ConnectionsTable(net_file_namebase))
    placer.shuffle_placement()
    print placer.evaluate_placement()
    schedule = VPRSchedule(placer.s2p, inner_num, placer.block_count, placer,
                           draw_enabled=draw_enabled)
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
        output_path = str(output_path)
    else:
        output_file_name = 'placed-%s-s%d-%s.h5' % (net_file_namebase, seed,
                                                    block_positions_sha1)
        if output_dir is not None:
            output_path = str(output_dir.joinpath(output_file_name))
        else:
            output_path = output_file_name

    parent_dir = path(output_path).parent
    if parent_dir and not parent_dir.isdir():
        parent_dir.makedirs_p()
    print 'writing output to: %s' % output_path

    h5f = ts.openFile(output_path, mode='w', filters=filters)

    net_file_results = h5f.createGroup(h5f.root, net_file_namebase,
                                       title='Placement results for %s CAMIP '
                                       'with `inner_num`=%s, wire-length '
                                       'driven' % (net_file_namebase,
                                                   inner_num))

    placements = h5f.createTable(
        net_file_results, 'placements',
        get_PLACEMENT_TABLE_LAYOUT(len(block_positions)),
        title='Placements for %s VPR' % net_file_namebase)
    placements.setAttr('net_file_namebase', net_file_namebase)

    placements.cols.block_positions_sha1.createIndex()
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

    stats_group = h5f.createGroup(net_file_results, 'placement_stats',
                                  title='Placement statistics for each '
                                  'outer-loop iteration of a anneal for %s' %
                                  net_file_namebase)

    # Prefix `block_positions_sha1` with `P_` to ensure the table-name is
    # compatible with Python natural-naming.  This is necessary since SHA1
    # hashes may start with a number, in which case the name would not be a
    # valid Python attribute name.
    placement_stats = h5f.createTable(stats_group, 'P_' + block_positions_sha1,
                                      get_PLACEMENT_STATS_TABLE_LAYOUT(),
                                      title='Placement statistics for each '
                                      'outer-loop iteration of a VPR anneal '
                                      'for %s with which produced the '
                                      'block-positions with SHA1 hash `%s`'
                                      % (net_file_namebase,
                                         block_positions_sha1))
    placement_stats.setAttr('net_file_namebase', net_file_namebase)
    placement_stats.setAttr('block_positions_sha1', block_positions_sha1)

    for i, stats in place_stats.iterrows():
        stats_row = placement_stats.row
        for field in ('start', 'end', 'temperature', 'cost', 'success_ratio',
                      'radius_limit', 'total_iteration_count'):
            stats_row[field] = stats[field]
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
    parser.add_argument('-e', '--draw-enabled', action='store_true')
    parser.add_argument('-i', '--inner-num', type=float, default=1.)
    parser.add_argument('-t', '--timing', action='store_true')
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
                                args.inner_num, args.timing, args.draw_enabled)

    extract_positions(placer.block_slot_keys, placer.p_x, placer.p_y,
                      placer.s2p)
    p_z = np.zeros(placer.p_x.size, dtype=np.uint32)
    p_z[:placer.s2p.io_count] = (placer.block_slot_keys[:placer.s2p.io_count] %
                                 placer.io_capacity)
    block_positions = np.array([placer.p_x[:], placer.p_y[:], p_z],
                               dtype='uint32').T
    save_placement(args.net_file_namebase, block_positions, place_stats,
                   output_path=args.output_path, output_dir=args.output_dir,
                   inner_num=args.inner_num, seed=args.seed)
    if args.draw_enabled:
        raw_input()
