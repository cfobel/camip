# coding: utf-8
from collections import OrderedDict
import hashlib

from path_helpers import path
import numpy as np
import pandas as pd
from fpga_netlist.connections_table import (ConnectionsTable,
                                            CONNECTION_DRIVER,
                                            CONNECTION_CLOCK_DRIVER,
                                            CONNECTION_SINK, INPUT_BLOCK,
                                            OUTPUT_BLOCK)
from si_prefix import si_format
from .bin.vpr_net_to_df import vpr_net_to_df
from .device.CAMIP import extract_positions
from .table_layouts import get_PLACEMENT_STATS_DATAFRAME_LAYOUT
from . import CAMIP, VPRSchedule


def place_from_multi_hdf(net_file_namebase, h5f_netlists_path, *args, **kwargs):
    connections_table = ConnectionsTable.from_net_list_name(net_file_namebase,
                                                            h5f_netlists_path)
    return place(connections_table, *args, **kwargs)


def place_from_hdf(h5f_netlist_path, *args, **kwargs):
    df_netlist = pd.read_hdf(str(h5f_netlist_path), '/connections')
    connections_table = ConnectionsTable(df_netlist)
    return place(connections_table, *args, **kwargs)


def place_from_vpr_net(vpr_net_path, *args, **kwargs):
    df_netlist = vpr_net_to_df(vpr_net_path)
    connections_table = ConnectionsTable(df_netlist)
    return place(connections_table, *args, **kwargs)


def place(connections_table, seed, io_capacity=3, inner_num=1.,
          include_clock=False, timing=False, critical_path_only=False,
          wire_length_factor=0.5, criticality_exp=10.):
    np.random.seed(seed)
    if timing or critical_path_only:
        from camip.timing import CAMIPTiming

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

    params = pd.DataFrame(OrderedDict([
        ('seed', seed),
        ('inner_num', schedule.inner_num),
        ('start_temperature', schedule.anneal_schedule.start_temperature),
        ('start_rlim', schedule.anneal_schedule.start_rlim),
        ('include_clock', include_clock)]), index=[0])

    if timing:
        params['wire_length_factor'] = wire_length_factor
        params['criticality_exp'] = criticality_exp

    print '\nRuntime: %ss' % si_format((place_stats.end -
                                        place_stats.start).sum())
    return schedule, placer, params, place_stats


def placer_to_block_positions_df(placer):
    # Extract block labels from connections data frame.
    df_block_labels = (placer._connections_table.connections
                       .loc[placer._connections_table.connections.type
                            .isin([CONNECTION_SINK, CONNECTION_CLOCK_DRIVER,
                                   CONNECTION_DRIVER]),
                            ['driver_key', 'sink_key', 'type', 'driver_type',
                             'sink_type', 'block_label']].copy())
    df_block_labels['block_key'] = df_block_labels['driver_key'].values
    df_block_labels['block_type'] = df_block_labels['driver_type'].values
    df_block_labels.loc[df_block_labels.type == CONNECTION_SINK,
                        'block_key'] = df_block_labels['sink_key']
    df_block_labels.loc[df_block_labels.type == CONNECTION_SINK,
                        'block_type'] = df_block_labels['sink_type']
    block_labels = df_block_labels.set_index('block_key')[['block_label', 'block_type']].drop_duplicates().sort_index()

    extract_positions(placer.block_slot_keys, placer.p_x, placer.p_y,
                      placer.s2p)
    block_positions = pd.DataFrame(OrderedDict([('x', placer.p_x[:]),
                                                ('y', placer.p_y[:])]),
                                   dtype='uint32')

    # Set `z` position for input and output blocks.
    block_positions['z'] = np.zeros(placer.p_x.size, dtype=np.uint32)
    block_positions['type'] = block_labels.block_type.values.astype('uint32')
    p_z = block_positions.loc[block_positions.type.isin([INPUT_BLOCK,
                                                         OUTPUT_BLOCK]), 'z']
    block_positions.loc[block_positions.type.isin([INPUT_BLOCK, OUTPUT_BLOCK]),
                        'z'] = (placer.block_slot_keys[:][p_z.index] %
                                placer.io_capacity)
    block_positions.index = block_labels['block_label']
    return block_positions


def block_positions_df_to_vpr(df_block_positions, extent, net_path, arch_path,
                              output_path):
    nx, ny = extent
    with open(output_path, 'wb') as output:
        output.write('''\
Netlist file: {net_path} Architecture file: {arch_path}
Array size: {nx} x {ny} logic blocks

#block name     x   y   subblk      block number
#----------     --  --  ------      ------------
'''.format(net_path=net_path, arch_path=arch_path, nx=nx, ny=ny))
        df_block_positions.to_string(buf=output, columns=list('xyz'),
                                     header=False, index_names=False)
        print >> output, ''


def save_placement(net_file_namebase, block_positions, params, place_stats,
                   output_path=None, output_dir=None):
    # Use a hash of the block-positions to name the HDF file.
    block_positions_sha1 = hashlib.sha1(block_positions[list('xyz')].values
                                        .astype('uint32').data).hexdigest()
    params_i = params.iloc[0]

    hdf_kwargs = dict(format='table', complib='zlib', complevel=6)
    if output_path is not None:
        context = dict(net_file_namebase=net_file_namebase,
                       seed=params_i['seed'],
                       block_positions_sha1=block_positions_sha1)
        output_path = str(output_path) % context
    else:
        output_file_name = 'placed-%s-s%d-%s.h5' % (net_file_namebase,
                                                    params_i['seed'],
                                                    block_positions_sha1)
        output_path = output_file_name
    if output_dir is not None:
        output_path = str(output_dir.joinpath(output_path))

    parent_dir = path(output_path).parent
    if parent_dir and not parent_dir.isdir():
        parent_dir.makedirs_p()
    print 'writing output to: %s' % output_path

    params.to_hdf(str(output_path), '/params', data_columns=params.columns,
                  **hdf_kwargs)
    block_positions.to_hdf(str(output_path), '/block_positions',
                           data_columns=block_positions.columns, **hdf_kwargs)
    place_stats.to_hdf(str(output_path), '/place_stats',
                       data_columns=place_stats.columns, **hdf_kwargs)
    return path(output_path)
