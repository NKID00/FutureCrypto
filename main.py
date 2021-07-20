from typing import Dict, Tuple
from datetime import datetime

import tensorflow as tf
import numpy as np


def load_one_dataset(name: str) -> tf.data.Dataset:
    print(f'load dataset {name} ...')
    return tf.data.Dataset.from_tensor_slices((
        np.load(f'./data/{name}.npz')['arr_0'],
    )).window(60*60*2, shift=60*60*2, drop_remainder=True)


def load_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    print('load dataset ...')
    btcusdt = load_one_dataset('BTCUSDT')
    ethusdt = load_one_dataset('ETHUSDT')
    ethbtc = load_one_dataset('ETHBTC')

    print('preprocess dataset ...')
    y = btcusdt.skip(1).map(lambda x: x.take(60))
    train = tf.data.Dataset.from_tensor_slices((
        {
            'btcusdt': btcusdt.take(len(y) - 10).batch(64).prefetch(1),
            'ethusdt': ethusdt.take(len(y) - 10).batch(64).prefetch(1),
            'ethbtc': ethbtc.take(len(y) - 10).batch(64).prefetch(1)
        },
        {
            'btcusdt': y.take(len(y) - 10).batch(64).prefetch(1),
            'ethusdt': y.take(len(y) - 10).batch(64).prefetch(1),
            'ethbtc': y.take(len(y) - 10).batch(64).prefetch(1)
        }
    ))
    test = tf.data.Dataset.from_tensor_slices((
        {
            'btcusdt': btcusdt.take(len(y)).skip(len(y) - 10).batch(64).prefetch(1),
            'ethusdt': ethusdt.take(len(y)).skip(len(y) - 10).batch(64).prefetch(1),
            'ethbtc': ethbtc.take(len(y)).skip(len(y) - 10).batch(64).prefetch(1)
        },
        {
            'btcusdt': y.skip(len(y) - 10).batch(64).prefetch(1),
            'ethusdt': y.skip(len(y) - 10).batch(64).prefetch(1),
            'ethbtc': y.skip(len(y) - 10).batch(64).prefetch(1)
        }
    ))
    return train, test


def save_model(model: tf.keras.Model):
    print('save model ...')
    model.save(f'./model/{datetime.now().strftime("%Y%m%dT%H%M%S")}')


def load_model(name) -> tf.keras.Model:
    print('load model ...')
    model = tf.keras.models.load_model(f'./model/{name}')
    model.summary()
    return model


def build_model() -> tf.keras.Model:
    print('build model ...')
    btcusdt = tf.keras.layers.Input(shape=[None, 1], name='btcusdt')
    lstm_1 = tf.keras.layers.LSTM(64, name='ltsm_1')(btcusdt)
    output_1 = tf.keras.layers.Dense(60, name='output_1')(lstm_1)

    ethusdt = tf.keras.layers.Input(shape=[None, 1], name='ethusdt')
    lstm_2 = tf.keras.layers.LSTM(64, name='ltsm_2')(ethusdt)

    ethbtc = tf.keras.layers.Input(shape=[None, 1], name='ethbtc')
    lstm_3 = tf.keras.layers.LSTM(64, name='ltsm_3')(ethbtc)

    concat_1 = tf.keras.layers.concatenate(
        [lstm_1, lstm_2, lstm_3], axis=-1, name='concat_1'
    )
    dense_1 = tf.keras.layers.Dense(64, name='dense_1')(concat_1)
    output_2 = tf.keras.layers.Dense(60, name='output_2')(dense_1)

    model = tf.keras.Model(
        inputs=[btcusdt, ethusdt, ethbtc],
        outputs=[output_1, output_2]
    )

    print('compile model ...')
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train_model(
    model: tf.keras.Model,
    train: tf.data.Dataset,
    test: tf.data.Dataset
):
    print('train model ...')
    model.fit(train, epochs=5)
    print('evaluate model ...')
    model.evaluate(test)


def main():
    model = build_model()
    train, test = load_dataset()
    train_model(model, train, test)
    save_model(model)


if __name__ == '__main__':
    main()
