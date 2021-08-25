import assist

import stumpy
import numpy as np

import pylab as plt
import tqdm


def key_to_label(k):
    return k.replace('max', '').strip('_').replace('__', r' \textrightarrow{} ', 1).replace('____', ' Host: ')


q = assist.build_query(select='MAX(rate)', from_='rpm', where="rate > 0 and L1 = '234.1.1.3'",
                       groupby=('time(10s) ', 'L1', 'L3', 'host', 'fill(0)'),
                       )
print(q)
df = assist.run_multivariate_query(q)
df.drop(columns='database', inplace=True)
df.dropna(how='all', inplace=True)
df.fillna(0, inplace=True)

print(df)

fig, axes = plt.subplots(2, 1)
plt.sca(axes[0])
for k in df.keys():
    plt.plot(df[k], label=k.replace('_', ' '))
plt.legend(loc=0)
plt.sca(axes[1])
for k in df.keys():
    plt.plot(df[k].to_numpy(), label=key_to_label(k))
plt.legend(loc=0)

_, axes1 = plt.subplots(len(df.keys()), 1, sharex=True)
_, axes2 = plt.subplots(len(df.keys()), 1, sharex=True)

m = 300
for ix, k in enumerate(tqdm.tqdm(df.keys())):
    opts = dict(alpha=0.5)

    plt.sca(axes1[ix])
    mp = stumpy.stump(df[k].to_numpy(), m)
    all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])

    plt.plot(df[k], color='k', alpha=0.1, linewidth=0.5)
    for i in range(unanchored_chain.shape[0]):
        y = df[k].iloc[unanchored_chain[i]:unanchored_chain[i] + m]
        x = y.index.values
        plt.plot(x, y)

    plt.sca(axes2[ix])
    plt.axis('off')
    for i in range(unanchored_chain.shape[0]):
        data = df[k].reset_index(drop=True).iloc[unanchored_chain[i]:unanchored_chain[i] + m].reset_index().values
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x - x.min() + int(m * 1.4) * i, y - y.mean())
plt.show()
