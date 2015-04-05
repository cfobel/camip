from collections import OrderedDict


def get_PLACEMENT_TABLE_LAYOUT(block_count):
    return OrderedDict([('net_list_name', ts.StringCol(250, pos=0)),
                        ('seed', ts.UInt32Col(pos=1)),
                        ('block_positions_sha1', ts.StringCol(40, pos=2)),
                        ('block_positions',
                         ts.UInt32Col(pos=2, shape=(block_count, 3))),
                        ('start', ts.Float64Col(pos=4)),
                        ('end', ts.Float64Col(pos=5)),
                        ('inner_num', ts.Float32Col(pos=6))])


def get_PLACEMENT_STATS_TABLE_LAYOUT():
    '''
    This table layout contains all fields printed in the summary table during a
    VPR _place_ operation.  It also includes two unix-timestamps, `start` and
    `end`, which mark the start-time and end-time of the corresponding
    outer-loop iteration, respectively.
    '''
    return OrderedDict([('start', ts.Float64Col(pos=1)),
                        ('end', ts.Float64Col(pos=2)),
                        ('temperature', ts.Float32Col(pos=3)),
                        ('cost', ts.Float64Col(pos=4)),
                        ('success_ratio', ts.Float32Col(pos=5)),
                        ('radius_limit', ts.Float32Col(pos=6)),
                        ('total_iteration_count', ts.UInt32Col(pos=7))])


def get_PLACEMENT_STATS_DATAFRAME_LAYOUT():
    return [('start', 'double'), ('end', 'double'), ('temperature', 'float'),
            ('cost', 'double'), ('success_ratio', 'float'),
            ('radius_limit', 'float'), ('evaluated_move_count', 'uint32')]
