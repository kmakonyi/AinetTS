import assist

import stumpy
import numpy as np

import pylab as plt


# NOTE: This takes a very long time, and doesn't yield anything interesting.

q = assist.build_query(select='MAX(rate)', from_='rpm', where='rate > 0',
                       groupby=('time(10s) ', 'L1', 'L3', 'host', 'fill(0)'),
                       )
print(q)
df = assist.run_multivariate_query(q)
df.drop(columns='database', inplace=True)
df.dropna(how='all', inplace=True)
df.fillna(0, inplace=True)

print(df)

print(df.dtypes)
print(df.memory_usage() * 1e-6)

m = 50
Ts = [df[k] for k in df.keys()]

radius, Ts_idx, subseq_idx = stumpy.ostinato(Ts, m)
print(f'Found Best Radius {np.round(radius, 2)} in time series {Ts_idx} starting at subsequence index location {subseq_idx}.')

seed_motif = Ts[Ts_idx][subseq_idx : subseq_idx + m]
x = np.linspace(0,1,50)
nn = np.zeros(len(Ts), dtype=np.int64)
nn[Ts_idx] = subseq_idx
for i, e in enumerate(Ts):
    if i != Ts_idx:
        nn[i] = np.argmin(stumpy.core.mass(seed_motif, e))
        lw = 1
        label = None
    else:
        lw = 4
        label = 'Seed Motif'
    plt.plot(x, e[nn[i]:nn[i]+m], lw=lw, label=label)
plt.title('The Consensus Motif')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
