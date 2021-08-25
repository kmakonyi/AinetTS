# Examples

All assume:

    import assist
    import pandas as pd 

A simple query:

     t0 = assist.Datetime(year=2021, month=6, day=6)
     t1 = assist.Datetime(year=2021, month=6, day=17, hour=23, minute=59)
     q = assist.build_query(select='100-MEAN(value)', from_='system_load',
                            where=f"time >= {t0} and time <= {t1} and host != \'HAU-AP1-WACS-01\' and L2=\'cpuload\' and \"name\" = \'Idle\'",
                            groupby=('time(10s)', 'host'))

A nested query:

    inner_q = build_query(select='time, value, host, L3', from_='system_load',
                          where='L2=\'cpuload\' and "name" != \'Idle\'',
                          )
    outer_q = build_query(select='time, value', from_=inner_q,
                          where='time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\'',
                          groupby=('host', 'L3'))

This raises a performance warning because time filtering is in the inner query, but works too:

    inner_q3 = build_query(select='time, value', from_='system_load',
                           where='time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and L2=\'cpuload\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))
    outer_q3 = build_query(select='time, value', from_=inner_q3)