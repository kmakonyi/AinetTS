import influxdb

import pandas as pd

client = influxdb.InfluxDBClient(database='frenymlab')


def run_query(query: str, *, verbose: bool = False) -> pd.DataFrame:
    data = []

    if verbose:
        print('Running query')
    response = client.query(query)

    if verbose:
        print('Combining data')
    for key in response.keys():
        database, tags = key
        this_df = pd.DataFrame(response.get_points(database, tags))
        this_df['database'] = database

        if tags:
            for ix, v in tags.items():
                this_df[ix] = v

        data.append(this_df)

    if verbose:
        print('Merging')
    data = pd.concat(data)
    if verbose:
        print('Parsing time')
    data['time'] = pd.to_datetime(data.time).dt.tz_convert(None)
    data.set_index('time', inplace=True)
    return data


def run_multivariate_query(query: str, separator: str = '__', *, verbose: bool = False) -> pd.DataFrame:
    data = []

    if verbose:
        print('Running query')
    response = client.query(query)
    databases = set(k[0] for k in response.keys())
    many_db = len(databases) > 1

    if verbose:
        print('Found data from', len(databases), 'database' + 's' if len(databases) > 1 else '')
        print('Combining data')

    for key in response.keys():
        database, tags = key
        this_df = pd.DataFrame(response.get_points(database, tags))

        if many_db:
            this_df['database'] = database

        if tags is None:
            tag_name = ''
        else:
            tag_name = separator.join(tags[k] for k in sorted(tags.keys()))

        columns = {key: separator.join((key, tag_name)) for key in this_df.keys() if key != 'time'}
        this_df.rename(columns=columns, inplace=True)
        this_df['time'] = pd.to_datetime(this_df.time).dt.tz_convert(None)
        this_df.set_index('time', inplace=True)

        data.append(this_df)

    if verbose:
        print('Merging')
    data = data[0].join(data[1:], how='outer')
    if not many_db:
        data['database'] = database
    return data
