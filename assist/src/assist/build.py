from __future__ import annotations

import warnings
import datetime


def build_query(select: str, from_: str, where: str = None, groupby: str | tuple | list = None, reverse: bool = False,
                limit: int = None, slimit: int = None, offset: int = None, soffset: int = None) -> str:
    """
    Build an InfluxDB query from its parts

    :param select: the fields
    :param from_: the measurement table or a sub-query
    :param where: optional filter
    :param groupby: optional grouping
    :param reverse: decreasing order
    :param limit: don't return more than this many points per series
    :param slimit: don't return more than this many series
    :param offset: pagination for each series
    :param soffset: pagination of individual series
    :return: the query
    """
    query = ['SELECT', select, 'FROM']

    from_lower = from_.lower()
    if 'select ' in from_lower and ' from ' in from_lower:
        # Nested query, wrap in parenthesis:
        from_ = f'( {from_} )'

        if 'where' in from_lower:
            if 'time' in from_lower.split('where')[1]:
                warnings.warn(RuntimeWarning('It seems you are filtering by time in the nested query.'
                                             'Consider moving it to the outermost query for better performance.'))

    query.append(from_)

    if where:
        query.extend(('WHERE', where))

    if groupby:
        query.append('GROUP BY')
        if isinstance(groupby, str):
            query.append(groupby)
        elif isinstance(groupby, (list, tuple)):
            # We need to wrap the tags but not the time with quotes
            query.append(','.join(gr if 'time(' in gr else f'"{gr}"' for gr in groupby))
        else:
            raise ValueError('Unknown type of groupby:', groupby)

    if reverse:
        query.append('ORDER BY time DESC')

    if limit:
        query.append(f'LIMIT {limit:d}')

    if offset:
        if not limit:
            raise ValueError('OFFSET requires setting LIMIT')
        query.append(f'OFFSET {offset:d}')

    if slimit:
        query.append(f'SLIMIT {slimit}')

    if soffset:
        if not slimit:
            raise ValueError('SOFFSET requires setting SLIMIT')
        query.append(f'SOFFSET {soffset:d}')

    return ' '.join(query)


class Datetime(datetime.datetime):
    def __str__(self):
        seconds = self.second + 1e-6 * self.microsecond
        seconds_str = f'{int(seconds):02d}.' + str(seconds % 1).split('.')[1]

        return f"'{self.year}-{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}:{seconds_str}'"

