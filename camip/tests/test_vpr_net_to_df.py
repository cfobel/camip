import pkg_resources

import pandas as pd
from pandas.util.testing import assert_frame_equal
from camip.bin.vpr_net_to_df import vpr_net_to_df
from path_helpers import path


def test_vpr_net_to_df():
    vpr_net_path = pkg_resources.resource_filename('camip',
                                                   path('tests')
                                                   .joinpath('fixtures',
                                                             '13blocks.net'))
    df_netlist = vpr_net_to_df(vpr_net_path)
    hdf_net_path = pkg_resources.resource_filename('camip',
                                                   path('tests')
                                                   .joinpath('fixtures',
                                                             '13blocks.h5'))
    df_netlist_fixture = pd.read_hdf(hdf_net_path, '/connections')
    assert_frame_equal(df_netlist, df_netlist_fixture)
