import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from path_helpers import path
from vpr_netfile_parser.VprNetParser import cVprNetFileParser


try:
    profile
except NameError:
    profile = lambda f: f


INPUT_DRIVER_PIN = 0
LOGIC_DRIVER_PIN = 4
LOGIC_BLOCK = 0
INPUT_BLOCK = 1
OUTPUT_BLOCK = 2
CLOCK_PIN = 5

CONNECTION_CLOCK = 5
CONNECTION_DRIVER = 200
CONNECTION_SINK = 100
CONNECTION_CLOCK_DRIVER = 30

# Connection type = DRIVER_TYPE + 10 * SINK_TYPE
DELAY_IO_TO_IO = INPUT_BLOCK + 10 * OUTPUT_BLOCK
DELAY_FB_TO_FB = LOGIC_BLOCK + 10 * LOGIC_BLOCK
DELAY_IO_TO_FB = INPUT_BLOCK + 10 * LOGIC_BLOCK
DELAY_FB_TO_IO = LOGIC_BLOCK + 10 * OUTPUT_BLOCK


@profile
def vpr_net_to_df(net_file_path):
    parser = cVprNetFileParser(net_file_path)

    block_labels = pd.Series(parser.block_labels)
    net_labels = pd.Series(parser.net_labels)
    type_labels = pd.Series(['.clb', '.input', '.output'],
                            index=[LOGIC_BLOCK, INPUT_BLOCK,
                                   OUTPUT_BLOCK])
    type_keys = pd.DataFrame(range(type_labels.shape[0]), dtype='uint32',
                             index=type_labels, columns=['type_key'])
    block_type_keys = type_keys.loc[parser.block_type,
                                    'type_key'].reset_index(drop=True)

    block_to_net_ids = parser.block_to_net_ids()
    net_key = np.concatenate(block_to_net_ids).astype('uint32')
    block_key = np.concatenate([[i] * len(v)
                                for i, v in
                                enumerate(block_to_net_ids)]).astype('uint32')
    pin_key = np.concatenate(parser.block_used_pins).astype('uint32')
    connections = pd.DataFrame(OrderedDict([('net_key', net_key),
                                            ('block_key', block_key),
                                            ('pin_key', pin_key)]))
    connections.insert(2, 'block_type',
                       block_type_keys.iloc[connections.block_key].values)
    connections['net_label'] = net_labels.iloc[connections.net_key].values
    connections['block_label'] = block_labels.iloc[connections.block_key].values
    return connections.sort(['net_key', 'block_key']).reset_index(drop=True)


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Convert VPR netlist `.net` file to HDF '
                            'connections `.h5` format.')
    parser.add_argument(dest='vpr_net_file', type=path)
    parser.add_argument(dest='hdf_file', type=path)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    df_netlist = vpr_net_to_df(args.vpr_net_file)
    df_netlist.to_hdf(str(args.hdf_file), '/connections', format='table',
                      data_columns=df_netlist.columns, complib='zlib',
                      complevel=6)
