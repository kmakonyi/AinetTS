import unittest.mock

import assist
import pytest


def test_simple_query():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    df = assist.run_query(q, cache=False)
    assert not df.empty


def test_nested_query():
    inner_q = assist.build_query(select='time, value, host, L3', from_='system_load',
                                 where='L2=\'cpuload\' and "name" != \'Idle\'',
                                 )
    outer_q = assist.build_query(select='time, value', from_=inner_q,
                                 where='time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\'',
                                 groupby=('host', 'L3'))
    df = assist.run_query(outer_q, cache=False)
    assert not df.empty


def test_nested_query_with_datetime():
    inner_q = assist.build_query(select='time, value', from_='system_load',
                                 where='L2=\'cpuload\' and "name" != \'Idle\'',
                                 groupby=('host', 'L3'))
    outer_q = assist.build_query(select='time, value', from_=inner_q,
                                 where=f'time > {assist.Datetime(year=2021, month=6, day=16)}'
                                       f'and time < {assist.Datetime(year=2021, month=6, day=17)}',
                                 )

    df = assist.run_query(outer_q, cache=False)
    assert not df.empty


def test_warning():
    inner_q = assist.build_query(select='time, value', from_='system_load',
                                 where=f'time > {assist.Datetime(year=2021, month=6, day=16)}'
                                       f'and time < {assist.Datetime(year=2021, month=6, day=17)}'
                                       'and L2=\'cpuload\' and "name" != \'Idle\'',
                                 groupby=('host', 'L3'))
    with pytest.warns(RuntimeWarning):
        outer_q = assist.build_query(select='time, value', from_=inner_q, )

    df = assist.run_query(outer_q, cache=False)
    assert not df.empty


def test_time_grouping():
    q = assist.build_query(select='time, MAX(value)', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('time(10m)', 'host', 'L3'))

    df = assist.run_query(q, cache=False)
    assert not df.empty


def test_fill_values():
    q = assist.build_query(select='time, MEAN(value)', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('time(10m)', 'fill(0)', 'host', 'L3'))

    df = assist.run_query(q, cache=False)
    assert not df.empty


def test_cached_query():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    def _run_query(q):
        df = assist.run_query(q, cache=True)
        return df

    _run_query(q)
    # Invalidate the InfluxDB client, it should still work
    df = unittest.mock.patch('assist.parse.client', new=None)(_run_query)(q)

    assert not df.empty

def test_nocached_query():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    @unittest.mock.patch('assist.parse.client', new=None)
    def _run_query(q):
        df = assist.run_query(q, cache=False)
        return df

    # Invalidate the InfluxDB client, it should fail
    with pytest.raises(AttributeError):
        _run_query(q)


def test_cached_query_mv():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    def _run_query(q):
        df = assist.run_multivariate_query(q, cache=True)
        return df

    _run_query(q)
    # Invalidate the InfluxDB client, it should still work
    df = unittest.mock.patch('assist.parse.client', new=None)(_run_query)(q)

    assert not df.empty

def test_nocached_query_mv():
    q = assist.build_query(select='time, value', from_='system_load',
                           where='L2=\'cpuload\' and time > \'2021-06-16 00:00:00\' and time < \'2021-06-17 00:00:00\' and "name" != \'Idle\'',
                           groupby=('host', 'L3'))

    @unittest.mock.patch('assist.parse.client', new=list())
    def _run_query(q):
        df = assist.run_multivariate_query(q, cache=False)
        return df

    # Invalidate the InfluxDB client, it should fail
    with pytest.raises(AttributeError):
        _run_query(q)