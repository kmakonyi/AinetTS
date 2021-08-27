import pprint

import assist

import numpy as np
import tensorflow as tf
from scipy import signal

import matplotlib.pyplot as plt

import tqdm


def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=50)
    ])


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_data, val_data, label_columns=None):
        # Store the raw data.
        self.train_df = train_data
        self.val_df = val_data

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)


def train_model(window_generator, patience=2, MAX_EPOCHS=20):
    model = get_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    model.fit(window_generator.train, epochs=MAX_EPOCHS,
              validation_data=window_generator.val,
              callbacks=[early_stopping], verbose=0)
    return model


t0 = assist.Datetime(year=2021, month=6, day=16, hour=1, minute=13, second=20)
t1 = assist.Datetime(year=2021, month=6, day=18, hour=1, minute=13, second=20)
q = assist.build_query(select='MAX(rate)', from_='rpm',
                       where=f'rate > 0 and time > {t0} and time < {t1}',
                       groupby=('L1', 'L3', 'host', 'time(10s)', 'fill(0)'),
                       )
print(q)

df = assist.run_multivariate_query(q)
df.drop(columns='database', inplace=True)

n = len(df)
column_indices = {name: i for i, name in enumerate(df.columns)}

cutoff = int(n * 0.7)

train_df = df[0:cutoff]
val_df = df[cutoff:]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std

norm_df = (df - train_mean) / train_std
models = {}

keys = list(df.keys())

input_width = 400
label_width = 50
for k in tqdm.tqdm(keys, desc='Training models'):
    windower = WindowGenerator(input_width=input_width, label_width=label_width, shift=label_width,
                               train_data=train_df[[k]], val_data=val_df[[k]],
                               label_columns=[k])

    model = train_model(windower)
    models[k] = model

performance = {}

all_errors = {}
for ix, k in enumerate(tqdm.tqdm(keys, desc='Predicting')):
    model = models[k]

    windows = []
    labels = []
    indices = []
    series = norm_df[k]
    for i in range(cutoff + 1 - label_width, len(norm_df) - input_width - label_width):
        windows.append(series.iloc[i:i + input_width])
        labels.append(series.iloc[i + input_width: i + input_width + label_width])
        indices.append(series.index.to_series().iloc[i + input_width + label_width])

    windows = np.array(windows)
    labels = np.array(labels)
    indices = np.array(indices)
    pred = model.predict(windows)

    error = (labels - pred).mean(axis=1)
    point_error = labels[:, -1] - pred[:, -1]

    plt.figure(1)
    plt.hist(point_error, histtype='step')
    plt.title('Error in the last point')

    plt.figure(2)
    plt.hist(labels.flatten() - pred.flatten(), histtype='step')
    plt.title('Errors in the full forecast')

    all_errors[k] = error / np.abs(error).max()

    plt.figure(3)
    line, = plt.plot(indices, np.abs(error), alpha=0.5)

    plt.figure(4)
    plt.plot(indices, signal.savgol_filter(np.abs(error), window_length=21, polyorder=3), alpha=0.3)
    plt.title('Absolute value of the error')

    performance[k] = line.get_color(), np.mean(np.abs(error)), np.mean(np.abs(point_error))

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(indices, labels[:, -1], label='True', alpha=0.5)
    axes[0].plot(indices, pred[:, -1], label='Predicted', alpha=0.5)
    axes[0].legend(loc='best')

    axes[1].plot(indices, signal.savgol_filter(np.abs(error), window_length=21, polyorder=3))
    axes[1].plot(indices, signal.savgol_filter(np.abs(point_error), window_length=21, polyorder=3), color='b',
                 alpha=0.1)
    axes[1].set_ylabel('Anomaly')
    axes[1].set_xlabel('Time')
    plt.suptitle(k.replace('max', '').strip('_').replace('__', r' \textrightarrow{} ', 1).replace('__', ': ').strip())

pprint.pprint(performance)

plt.figure()
for color, base, point in performance.values():
    plt.scatter(base, point, color=color)

plt.figure()
all_errors_matrix = np.stack(list(all_errors.values()))
M = np.abs(all_errors_matrix).max()
plt.imshow(all_errors_matrix, vmin=-M, vmax=M, cmap='bwr', interpolation='nearest', aspect='auto')
plt.yticks(range(len(keys)),
           [k.replace('max', '').strip('_').replace('__', r' \textrightarrow{} ', 1).replace('__', ': ').strip()
            for k in keys])

STEP = len(indices) // 5
plt.xticks(np.arange(len(indices), step=STEP), indices[::STEP])
plt.title('Forecast error')
plt.xlabel('Time')
plt.ylabel('Bitrate')
plt.savefig('examples/forecast_error.png')

plt.imshow(np.abs(all_errors_matrix), interpolation='nearest', aspect='auto')
plt.yticks(range(len(keys)),
           [k.replace('max', '').strip('_').replace('__', r' \textrightarrow{} ', 1).replace('__', ': ').strip()
            for k in keys])

STEP = len(indices) // 5
plt.xticks(np.arange(len(indices), step=STEP), indices[::STEP])
plt.title('Forecast error')
plt.xlabel('Time')
plt.ylabel('Bitrate')
plt.savefig('examples/forecast_error_abs.png')
plt.show()
