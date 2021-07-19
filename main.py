from typing import Tuple
from datetime import datetime

import tensorflow as tf
import numpy as np


def load_one_dataset(name: str) -> tf.data.Dataset:
    print(f'load dataset {name} ...')
    array = np.load(f'./data/{name}.npz')['arr_0']
    dataset = tf.data.Dataset.from_tensor_slices((array,))
    return dataset.window(60*60*2, shift=60*60*2, drop_remainder=True)


def load_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    print('load dataset ...')
    btcusdt_dataset = load_one_dataset('BTCUSDT')
    ethusdt_dataset = load_one_dataset('ETHUSDT')
    ethbtc_dataset = load_one_dataset('ETHBTC')
    print('preprocess dataset ...')
    dataset = tf.data.Dataset.zip((
        btcusdt_dataset, ethusdt_dataset, ethbtc_dataset
    ))
    dataset = tf.data.Dataset.zip((
        dataset,
        btcusdt_dataset.skip(1).map(lambda x: x.take(60))
    ))
    train_dataset = (
        dataset.take(len(dataset) - 10)
        .shuffle(100)
        .batch(64)
    )
    test_dataset = dataset.skip(len(dataset) - 10).batch(64)
    return train_dataset, test_dataset


def save_model(model: tf.keras.Model):
    print('save model ...')
    model.save(f'./model/{datetime.now().strftime("%Y%m%dT%H%M%S")}')


def load_model(name) -> tf.keras.Model:
    print('load model ...')
    return tf.keras.models.load_model(f'./model/{name}')


def main():
    print('build model ...')
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[None, 3]),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(60)
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    train_dataset, test_dataset = load_dataset()
    print('train model ...')
    model.fit(train_dataset, epochs=5)
    print('evaluate model ...')
    model.evaluate(test_dataset)
    save_model(model)


if __name__ == '__main__':
    main()
