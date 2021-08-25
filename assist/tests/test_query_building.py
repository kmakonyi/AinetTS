import assist
import pytest


def test_simple_query():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    df = assist.run_query(q)
    assert not df.empty


def test_nested_query():
    inner_q = assist.build_query(select='time, value, host, L3', from_='system_load',
                                 where='L2=\'cpuload\' and "name" != \'Idle\'',
                                 )
    outer_q = assist.build_query(select='time, value', from_=inner_q,
                                 where='time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\'',
                                 groupby=('host', 'L3'))
    df = assist.run_query(outer_q)
    assert not df.empty


def test_nested_query_with_datetime():
    inner_q = assist.build_query(select='time, value', from_='system_load',
                                 where='L2=\'cpuload\' and "name" != \'Idle\'',
                                 groupby=('host', 'L3'))
    outer_q = assist.build_query(select='time, value', from_=inner_q,
                                 where=f'time > {assist.Datetime(year=2021, month=6, day=16)}'
                                       f'and time < {assist.Datetime(year=2021, month=6, day=17)}',
                                 )

    df = assist.run_query(outer_q)
    assert not df.empty


def test_warning():
    inner_q = assist.build_query(select='time, value', from_='system_load',
                                  where=f'time > {assist.Datetime(year=2021, month=6, day=16)}'
                                        f'and time < {assist.Datetime(year=2021, month=6, day=17)}'
                                        'and L2=\'cpuload\' and "name" != \'Idle\'',
                                  groupby=('host', 'L3'))
    with pytest.warns(RuntimeWarning):
        outer_q = assist.build_query(select='time, value', from_=inner_q, )

    df = assist.run_query(outer_q)
    assert not df.empty
