# coding: utf-8
import hashlib
import sys
from collections import OrderedDict
import subprocess
import pkg_resources

from path_helpers import path
from table_layouts import (get_PLACEMENT_TABLE_LAYOUT,
                           get_PLACEMENT_STATS_TABLE_LAYOUT,
                           get_PLACEMENT_STATS_DATAFRAME_LAYOUT)
from camip import CAMIP, VPRSchedule, Placement, partition_keys
from camip.timing import CAMIPTiming
from camip.device.CAMIP import extract_positions
import numpy as np
import tables as ts
import pandas as pd
from cyplace_experiments.data.connections_table import (ConnectionsTable,
                                                        get_connections_frame,
                                                        populate_connection_frame,
                                                        LOGIC_BLOCK,
                                                        CONNECTION_CLOCK,
                                                        CONNECTION_CLOCK_DRIVER)


def get_version_info():
    try:
        version = subprocess.check_output('git describe', shell=True).strip()
        try:
            subprocess.check_call('git diff-files --quiet', shell=True)
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
        if dirty:
            version += '*'
    except:
        version = pkg_resources.get_distribution('camip').version
    return version


def result_dataframes(params, block_mapping, place_stats):
    stats_params_df = pd.DataFrame([params.values()], columns=params.keys(),
                                   index=range(place_stats.shape[0]))
    stats_params_df['state_index'] = np.arange(stats_params_df.shape[0],
                                               dtype=int)
    states_df = stats_params_df.join(place_stats)
    states_df['evaluated_count'] = 0.
    states_df['evaluated_count'].values[0] = (states_df
                                              ['total_iteration_count']
                                              .values[0])
    states_df['evaluated_count'].values[1:] = (
        states_df['total_iteration_count'].values[1:] -
        states_df['total_iteration_count'].values[:-1])
    states_df.drop('total_iteration_count', axis=1, inplace=True)

    block_params_df = pd.DataFrame([params.values()], columns=params.keys(),
                                index=range(block_mapping.shape[0]))

    return states_df, block_params_df.join(block_mapping)


def place(net_file_namebase, seed, io_capacity=3, inner_num=1.,
          include_clock=False, timing=False, critical_path_only=False,
          wire_length_factor=0.5, criticality_exp=10.):
    connections = populate_connection_frame(
        get_connections_frame(net_file_namebase))
    connections['original_block_key'] = connections['block_key']
    if include_clock:
        place_connections = connections
    else:
        connections['exclude'] = connections.type.isin(
            [CONNECTION_CLOCK, CONNECTION_CLOCK_DRIVER])
        place_connections = partition_keys(connections, drop_exclude=True)

    # __NB__ We must create `Placement` _after_ calling `partition_keys`, since
    # it will likely reassign the block keys.
    placement = Placement(connections, io_capacity)
    print 'seed:', seed
    np.random.seed(seed)
    placement.shuffle()

    if timing or critical_path_only:
        if timing:
            critical_path_only = False
        sync_logic_block_keys = (
            connections.loc[(connections.block_type == LOGIC_BLOCK) &
                            (connections.type ==
                             CONNECTION_CLOCK)]['block_key'].unique()
            .astype(np.int32))
        #placer = CAMIPTiming(place_connections, placement,
        placer = CAMIPTiming(connections, placement,
                             sync_logic_block_keys,
                             timing_cost_disabled=critical_path_only,
                             wire_length_factor=wire_length_factor,
                             criticality_exp=criticality_exp)
    else:
        placer = CAMIP(place_connections, placement)

    print placer.evaluate_placement()
    schedule = VPRSchedule(placer.s2p, inner_num, placer.block_count, placer)
    print 'starting temperature: %.2f' % schedule.anneal_schedule.temperature

    states = schedule.run(placer)

    # Convert list of state dictionaries into a `pandas.DataFrame`.
    stats_layout = pd.DataFrame(
        np.array([], dtype=get_PLACEMENT_STATS_DATAFRAME_LAYOUT()))
    place_stats = stats_layout.append(states)

    placement.block_data.v['slot_key'][:placer.block_count] = \
        placer.block_data['slot_key'][:]
    block_positions = placement.block_positions()[:].values.astype(np.uint32)
    block_mapping = (connections[['block_key', 'original_block_key',
                                'block_label']].drop_duplicates('block_key')
                     .sort('original_block_key'))
    positions = block_positions[block_mapping['block_key']]
    block_mapping['x'] = positions[:, 0]
    block_mapping['y'] = positions[:, 1]
    block_mapping['z'] = positions[:, 2]
    placement_sha1 = (hashlib.sha1(block_mapping[['x', 'y', 'z']].values.data)
                      .hexdigest())

    version = get_version_info()
    params = OrderedDict([('version', version),
                          ('net_file_namebase', net_file_namebase),
                          ('io_capacity', io_capacity),
                          ('inner_num', inner_num),
                          ('include_clock', include_clock),
                          ('timing', timing),
                          ('critical_path_only', critical_path_only),
                          ('wire_length_factor', wire_length_factor),
                          ('max_criticality_exp', criticality_exp),
                          ('seed', seed),
                          ('placement_sha1', placement_sha1)])
    states_df, positions_df = result_dataframes(params, block_mapping,
                                                place_stats)
    return params, states_df, positions_df


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
    parser.add_argument('-f', '--force-overwrite', action='store_true')

    args = parser.parse_args()
    if args.output_path is not None and (args.output_path.isfile() and
                                         not args.force_overwrite):
        parser.error('Output path exists.  Use `-f` to force overwrite.')
    return args


if __name__ == '__main__':
    args = parse_args()
    print args

    np.random.seed(args.seed)
    params, states_df, positions_df = place(args.net_file_namebase, args.seed,
                                            args.io_capacity, args.inner_num,
                                            args.include_clock, args.timing,
                                            args.critical_path,
                                            args.wire_length_factor,
                                            args.criticality_exp)

    short_params = OrderedDict([('seed', ('s', '%d')),
                                ('io_capacity', ('I', '%d')),
                                ('inner_num', ('i', '%.2f')),
                                ('include_clock', ('C', None)),
                                ('timing', ('t', None)),
                                ('critical_path_only', ('c', None)),
                                ('wire_length_factor', ('w', '%.2f')),
                                ('max_criticality_exp', ('e', '%.2f'))])
    if not args.timing:
        del short_params['timing']
        del short_params['wire_length_factor']
        del short_params['max_criticality_exp']
    if args.timing or not args.critical_path:
        del short_params['critical_path_only']
    if not args.include_clock:
        del short_params['include_clock']

    # Convert parameter to short string representation.
    extract_param = lambda x, k, v: v[0] + (v[1] % x[k]) if v[1] is not None else v[0]

    if args.output_path is None:
        params_str = '-'.join([positions_df.iloc[0][k]
                               for k in ('net_file_namebase',
                                         'placement_sha1')] +
                              [extract_param(positions_df.iloc[0], k, v)
                               for k, v in short_params.items()])
        args.output_path = 'placed-%s.h5' % params_str
        if args.output_dir is not None:
            args.output_path = args.output_dir.joinpath(args.output_path)

    h5f = pd.HDFStore(str(args.output_path), 'w')
    h5f.put('/states', states_df, format='table', complevel=2, complib='zlib')
    h5f.put('/positions', positions_df, format='table', complevel=2,
            complib='zlib')
    h5f.close()
