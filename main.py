from typing import Tuple
from datetime import datetime
from os import cpu_count, walk

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


SUBSAMPLING_SIZE = 15
INPUT_SIZE = 60*60*24 // SUBSAMPLING_SIZE  # 1 day
OUTPUT_SIZE = 60*60*12 // SUBSAMPLING_SIZE  # 0.5 day
TEST_DATASET_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100
BATCH_SIZE = cpu_count()


def load_preprocess_array(name: str) -> Tuple[
    np.ndarray, np.ndarray, Tuple[np.float64, np.float64]
]:
    print(f'load array {name} ...')
    arr = np.load(f'./data/{name}.npz')['arr_0']
    print(f'preprocess array {name} ...')
    xp = (arr.min(), arr.max())
    arr = arr[::SUBSAMPLING_SIZE]  # subsampling
    # scale the values to [-1.0, +1.0]
    arr = np.interp(arr, xp, (-1.0, +1.0))
    # split into chunks of INPUT_SIZE
    arr = arr[:len(arr) // INPUT_SIZE * INPUT_SIZE]
    arr = arr.reshape(len(arr) // INPUT_SIZE, INPUT_SIZE)
    # the first 1 output_size for every input_size after 1 input_size
    y = arr[1:][:, :OUTPUT_SIZE]
    arr = arr[:-1]
    arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
    return arr, y, xp
    

def load_dataset() -> Tuple[
    tf.data.Dataset, tf.data.Dataset, Tuple[np.float64, np.float64]
]:
    print('load and preprocess array ...')
    btcusdt, y, xp = load_preprocess_array('BTCUSDT')
    ethusdt = load_preprocess_array('ETHUSDT')[0]
    ethbtc = load_preprocess_array('ETHBTC')[0]

    print('build dataset ...')
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'btcusdt': btcusdt, 'ethusdt': ethusdt, 'ethbtc': ethbtc,
        },
        {
            'output_1': y, 'output_2': y
        }
    )).take(len(y))
    train = (
        dataset.take(len(y) - TEST_DATASET_SIZE)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )
    test = (
        dataset.skip(len(y) - TEST_DATASET_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )
    return train, test, xp


def save_model(model: tf.keras.Model):
    print('save model ...')
    name = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f'model name = {name}')
    model.save(f'./model/{name}')


def load_model() -> tf.keras.Model:
    print('load model ...')
    name = sorted(next(walk("model"))[1])[-1]
    print(f'model name = {name}')
    model = tf.keras.models.load_model(f'./model/{name}')
    model.summary()
    return model


def build_model() -> tf.keras.Model:
    print('build model ...')
    btcusdt = tf.keras.layers.Input(shape=[INPUT_SIZE, 1], name='btcusdt')
    lstm_1 = tf.keras.layers.LSTM(128, name='ltsm_1')(btcusdt)
    output_1 = tf.keras.layers.Dense(OUTPUT_SIZE, name='output_1')(lstm_1)

    ethusdt = tf.keras.layers.Input(shape=[INPUT_SIZE, 1], name='ethusdt')
    lstm_2 = tf.keras.layers.LSTM(128, name='ltsm_2')(ethusdt)

    ethbtc = tf.keras.layers.Input(shape=[INPUT_SIZE, 1], name='ethbtc')
    lstm_3 = tf.keras.layers.LSTM(128, name='ltsm_3')(ethbtc)

    concat_1 = tf.keras.layers.concatenate(
        [lstm_1, lstm_2, lstm_3], axis=-1, name='concat_1'
    )
    dense_1 = tf.keras.layers.Dense(128, name='dense_1')(concat_1)
    output_2 = tf.keras.layers.Dense(
        OUTPUT_SIZE, name='output_2'
    )(dense_1)

    model = tf.keras.Model(
        inputs=[btcusdt, ethusdt, ethbtc],
        outputs=[output_1, output_2]
    )

    print('compile model ...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            'output_1': tf.keras.losses.MeanSquaredError(),
            'output_2': tf.keras.losses.MeanSquaredError()
        },
        loss_weights={
            'output_1': 0.4,
            'output_2': 0.6
        }
    )
    model.summary()
    return model


def train_model(model: tf.keras.Model, train: tf.data.Dataset):
    print('train model ...')
    model.fit(train, epochs=5, verbose=1)


def evaluate_model(
    model: tf.keras.Model,
    test: tf.data.Dataset, xp: Tuple[np.float64, np.float64]
):
    print('evaluate model ...')
    model.evaluate(test, verbose=1)
    output = model.predict(test, verbose=1)
    test_data = next(iter(test))
    x_iter = test_data[0]['btcusdt']
    y_iter = test_data[1]['output_1']
    axes = plt.subplots(4, 2)[1]
    for ax, x, y, output_1, output_2 in zip(
        axes.flatten(), x_iter, y_iter, output[0], output[1]
    ):
        plot_prediction(
            ax,
            np.interp(x.numpy().flatten(), [-1.0, +1.0], xp),
            np.interp(y.numpy().flatten(), [-1.0, +1.0], xp),
            np.interp(output_1.flatten(), [-1.0, +1.0], xp),
            np.interp(output_2.flatten(), [-1.0, +1.0], xp)
        )
    plt.show()


def plot_one(ax, y, c, l):
    ax.plot(range(len(y), len(y) * 2), y, c=c, label=l)
    ax.scatter(range(len(y), len(y) * 2), y, c=c)


def plot_prediction(ax, x, y, output_1, output_2):
    ax.plot(
        range(len(y) * 2), np.concatenate((x[-len(y):], y)),
        c='gray', label='real', zorder=1
    )
    ax.scatter(
        range(len(y) * 2), np.concatenate((x[-len(y):], y)),
        c='gray', zorder=1
    )
    plot_one(ax, output_1, 'deepskyblue', 'output_1')
    plot_one(ax, output_2, 'blue', 'output_2')
    ax2 = ax.twinx()
    plot_one(ax2, output_1 - y, 'salmon', 'difference_1')
    plot_one(ax2, output_2 - y, 'red', 'difference_2')


def main():
    build_new_model = input('build new model? (y/n): ').lower().startswith('y')
    if build_new_model:
        model = build_model()
    else:
        model = load_model()
    train, test, xp = load_dataset()
    if build_new_model:
        train_model(model, train)
        save_model(model)
    evaluate_model(model, test, xp)


if __name__ == '__main__':
    main()
