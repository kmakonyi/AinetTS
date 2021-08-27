import assist

import numpy as np
import pylab as plt
from scipy import signal
import stumpy


def find_segmentation(series, window_size):
    mp = stumpy.stump(series.value.astype(np.float64), m=window_size)

    cac, regime_locations = stumpy.fluss(mp[:, 1], L=window_size, n_regimes=4, excl_factor=1)
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].plot(range(series.value.shape[0]), series.value)

    axs[1].plot(range(cac.shape[0]), cac, color='C1')

    for x in regime_locations:
        axs[0].axvline(x=x, linestyle="dashed")
        axs[1].axvline(x=x, linestyle="dashed")
    print(cac.shape, series.shape)


q = assist.build_query(select='MAX(rate)', from_='rpm',
                       where='rate > 0 and L1 = \'234.1.1.3\' and L3 = \'10.30.4.2\'',
                       groupby=('L1', 'L3', 'host', 'time(10s)', 'fill(0)'),
                       )

print(q)

df = assist.run_query(q)
df.dropna(how='any', inplace=True)

df.rename(columns={'max': 'value'}, inplace=True)
df['ix'] = np.arange(0, len(df))
ix = df.query('value > 0').tail(1).ix.to_list()[0]
df = df.query('ix <= @ix').copy()

series_1 = df.query('host == "R2"')
series_2 = df.query('host == "R4"')

# series_1.index = pd.DatetimeIndex(series_1.index, freq='infer')

autocorr = np.array([series_1.value.autocorr(lag) for lag in range(0, 310)])

peaks = signal.argrelmax(autocorr, order=5)[0]
prominences = signal.peak_prominences(autocorr, peaks)[0]
peaks = peaks[prominences > 2 * prominences.mean()]

for window_size in peaks:
    find_segmentation(series_1, window_size)
    plt.suptitle(f'Host R2, window = {window_size}')
    find_segmentation(series_2, window_size)
    plt.suptitle(f'Host R4, window = {window_size}')

plt.show()
