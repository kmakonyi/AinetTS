import pprint

import tensorflow as tf

import assist

import numpy as np
import pylab as plt

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.set_soft_device_placement(True)

t0 = assist.Datetime(year=2021, month=6, day=16, hour=1, minute=13, second=20)
t1 = assist.Datetime(year=2021, month=6, day=18, hour=1, minute=13, second=20)
q = assist.build_query(select='MAX(rate)', from_='rpm',
                       where=f'rate > 0 and L1 = \'234.1.1.3\' and L3 = \'10.30.4.2\' and host = \'R2\' and time > {t0} and time < {t1}',
                       groupby=('L1', 'L3', 'host', 'time(10s)', 'fill(0)'),
                       )
print(q)

df = assist.run_query(q)

df = df[['max']]

n = len(df)
column_indices = {name: i for i, name in enumerate(df.columns)}

train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

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

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

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

    def plot(self, model=None, plot_col='max', max_subplots=3, from_='train', force=True):
        if force:
            inputs, labels = next(iter(getattr(self, from_)))
        else:
            inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)

                plt.scatter(self.label_indices, predictions[n, :],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time')

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


w1 = WindowGenerator(input_width=400, label_width=50, shift=50,
                     label_columns=['max'])

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['max'])


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


baseline = Baseline(label_index=column_indices['max'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

train_performance = {}
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
train_performance['Baseline'] = baseline.evaluate(single_step_window.train, verbose=0)

dense = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(w1.input_width,)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=50)
])

dense.summary()
compile_and_fit(dense, w1)

val_performance['Dense'] = dense.evaluate(w1.val)
performance['Dense'] = dense.evaluate(w1.test, verbose=0)
train_performance['Dense'] = dense.evaluate(w1.train, verbose=0)

dense_small = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(w1.input_width,)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=50)
])

dense_small.summary()
compile_and_fit(dense_small, w1)
val_performance['Dense small'] = dense_small.evaluate(w1.val)
performance['Dense small'] = dense_small.evaluate(w1.test, verbose=0)
train_performance['Dense small'] = dense_small.evaluate(w1.train, verbose=0)

rec_model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(16, input_shape=(400, 1,), return_sequences=True),
    tf.keras.layers.GRU(128, return_sequences=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=50)
])

compile_and_fit(rec_model, w1)
rec_model.summary()
val_performance['Recurrent'] = rec_model.evaluate(w1.val)
performance['Recurrent'] = rec_model.evaluate(w1.test, verbose=0)
train_performance['Recurrent'] = rec_model.evaluate(w1.train, verbose=0)

pprint.pprint(performance)
pprint.pprint(val_performance)
pprint.pprint(train_performance)
