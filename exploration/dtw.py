"""This example uses dynamic time warping to find a flexible alignment between two series."""

import assist

import tslearn.metrics
import numpy as np
from scipy import spatial

import pylab as plt

q = assist.build_query(select='MAX(rate)', from_='rpm', where="rate > 0 and L1 = '234.1.1.3'",
                       groupby=('time(10s) ', 'L1', 'L3', 'host', 'fill(0)'),
                       limit=10_000
                       )
print(q)
df = assist.run_multivariate_query(q)
df.drop(columns='database', inplace=True)
df.dropna(how='all', inplace=True)
df.fillna(0, inplace=True)

print(df)

print(df.dtypes)
print(df.memory_usage() * 1e-6)

keys = list(df.keys())
s_y1, s_y2 = df[keys[0]].to_numpy()[:, np.newaxis], df[keys[1]].to_numpy()[:, np.newaxis]



path, sim = tslearn.metrics.dtw_path(s_y1, s_y2)

plt.figure(figsize=(8, 8))
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

mat = spatial.distance.cdist(s_y1, s_y2)

ax_gram.imshow(mat, origin='lower')
ax_gram.axis("off")
ax_gram.autoscale(False)
ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
             linewidth=1., ls=':', alpha=0.5)

size = len(df)
ax_s_x.plot(np.arange(size), s_y2, "b-", linewidth=1.)
ax_s_x.axis("off")
ax_s_x.set_xlim((0, size - 1))

ax_s_y.plot(-s_y1, np.arange(size), "b-", linewidth=1.)
ax_s_y.axis("off")
ax_s_y.set_ylim((0, size - 1))

plt.savefig(f'dtw_path.png')


std = np.mean([s_y1.std(), s_y2.std()])
s_y1,s_y2 = (s_y1 - s_y1.mean()) /std, (s_y2 - s_y2.mean()) / std
size = len(s_y1)

gamma = 1.
alignment, sim = tslearn.metrics.soft_dtw_alignment(s_y1, s_y2, gamma)

plt.figure(figsize=(8, 8))
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_gram = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

ax_gram = plt.axes(rect_gram)
ax_s_x = plt.axes(rect_s_x, sharex=ax_gram)
ax_s_y = plt.axes(rect_s_y, sharey=ax_gram)


ax_gram.imshow(alignment, origin='lower')
ax_gram.axis("off")
ax_gram.autoscale(False)
#ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
#             linewidth=1., ls=':', alpha=0.5)


ax_s_x.plot(np.arange(size), s_y2, "b-", linewidth=1.)
ax_s_x.axis("off")
ax_s_x.set_xlim((0, size - 1))

ax_s_y.plot(-s_y1, np.arange(size), "b-", linewidth=1.)
ax_s_y.axis("off")
ax_s_y.set_ylim((0, size - 1))

plt.suptitle(rf"$\gamma = {gamma:.1f}$")

plt.savefig(f'dtw_path_{gamma}.png')
plt.show()
