import assist

import stumpy
import numpy as np

import pylab as plt
import tqdm

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

fig, axes = plt.subplots(2, 1)
plt.sca(axes[0])
for k in df.keys():
    plt.plot(df[k], label=k.replace('_', ' '))
plt.legend(loc=0)
plt.sca(axes[1])
for k in df.keys():
    plt.plot(df[k].to_numpy(), label=k.replace('_', ' '))
plt.legend(loc=0)


sample_sizes = np.logspace(1, 3, num=6).astype(int)
for k in tqdm.tqdm(df.keys()):
    opts = dict(alpha=0.5)
    fig, axes = plt.subplots(len(sample_sizes) + 1, 1, sharex=True)
    plt.suptitle(k.replace('max', '').strip('_').replace('__', r' \textrightarrow{} ', 1).replace('____', ' Host: '))
    axes[0].plot(df[k], **opts)
    axes[0].set_ylabel('Bitrate')

    for ix, n in enumerate(sample_sizes):
        m_prof = stumpy.stump(df[k].to_numpy(), n)
        total_length = m_prof.shape[0]
        offset = (len(df.index) - total_length) // 2

        axes[ix + 1].plot(df.index[offset: offset + total_length], m_prof[:, 0], **opts)
        axes[ix + 1].set_ylabel(f'n={n}')

plt.show()
