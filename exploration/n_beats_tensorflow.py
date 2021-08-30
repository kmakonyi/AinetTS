import warnings

import matplotlib.pyplot as plt
import numpy as np

from nbeats_keras.model import NBeatsNet

import tqdm

import assist

warnings.filterwarnings(action='ignore', message='Setting attributes')


def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


t0 = assist.Datetime(year=2021, month=6, day=16, hour=1, minute=13, second=20)
t1 = assist.Datetime(year=2021, month=6, day=18, hour=1, minute=13, second=20)
q = assist.build_query(select='MAX(rate)', from_='rpm',
                       where=f'rate > 0 and time > {t0} and time < {t1}',
                       groupby=('L1', 'L3', 'host', 'time(10s)', 'fill(0)'),
                       )
print(q)

df = assist.run_multivariate_query(q)
df.drop(columns='database', inplace=True)

print(df.head())

forecast_length = 50
backcast_length = 50
batch_size = 10

k = 'max__234.1.1.1__10.30.4.2__R2'
x, y = [], []
for epoch in range(backcast_length, len(df) - forecast_length):
    x.append(df[k].iloc[epoch - backcast_length:epoch])
    y.append(df[k].iloc[epoch:epoch + forecast_length])

x = np.array(x)
y = np.array(y)

# split train/test.
c = int(len(x) * 0.8)
x_train, y_train = x[:c], y[:c]
x_test, y_test = x[c:], y[c:]

# normalization.
norm_constant = np.max(x_train)
x_train, y_train = x_train / norm_constant, y_train / norm_constant
x_test, y_test = x_test / norm_constant, y_test / norm_constant

# model
net = NBeatsNet(
    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
    forecast_length=forecast_length,
    backcast_length=backcast_length,
    hidden_layer_units=128,
)
net.compile(loss='mse', optimizer='adam')

grad_step = 0
for epoch in tqdm.trange(1000):
    # train.
    train_loss = []
    for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
        grad_step += 1
        loss = net.train_on_batch(x_train_batch, y_train_batch)
        train_loss.append(loss)

    train_loss = np.mean(train_loss)

    # test
    forecast = net.predict(x_test).squeeze()
    test_loss = np.mean(np.square(forecast - y_test))
    p = forecast
    if epoch % 10 == 0:
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        plt.figure(1)
        for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
            ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
            plt.sca(axes.flatten()[plot_id])
            plt.grid()
            plot_scatter(range(0, backcast_length), xx, color='b')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        plt.show()

        print(f'epoch = {str(epoch).zfill(4)}, '
              f'grad_step = {str(grad_step).zfill(6)}, '
              f'tr_loss (epoch) = {1000 * train_loss:.3f}, '
              f'te_loss (epoch) = {1000 * test_loss:.3f}')
