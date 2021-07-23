from typing import Tuple
from datetime import datetime

import tensorflow as tf
import numpy as np

from config import (
    INPUT_SIZE, OUTPUT_SIZE, TEST_DATASET_SIZE,
    SHUFFLE_BUFFER_SIZE, BATCH_SIZE, EPOCHS
)


def build_model() -> tf.keras.Model:
    print('build model ...')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=[INPUT_SIZE, 1]),
        tf.keras.layers.GRU(INPUT_SIZE * 3, return_sequences=True),
        tf.keras.layers.LSTM(INPUT_SIZE * 2),
        tf.keras.layers.Dense(INPUT_SIZE * 2, activation='tanh'),
        tf.keras.layers.Dense(OUTPUT_SIZE)
    ])

    print('compile model ...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanAbsoluteError()
    )
    model.summary()
    return model


def load_train_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    print('load data ...')
    data = np.load('./preprocessed/BTCUSDT.npz')
    arr = data['arr']
    # the first 1 output_size for every input_size 
    y = data['y'][:, :OUTPUT_SIZE]

    print('build dataset ...')
    dataset = tf.data.Dataset.from_tensor_slices((arr, y))
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
    return train, test


def train_model(
    model: tf.keras.Model, train: tf.data.Dataset, test: tf.data.Dataset
):
    print('train model ...')
    model.fit(train, epochs=EPOCHS, verbose=1)
    model.evaluate(test, verbose=1)


def save_model(model: tf.keras.Model):
    print('save model ...')
    name = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f'model name = {name}')
    model.save(f'./model/{name}')


if __name__ == '__main__':
    model = build_model()
    train, test = load_train_data()
    train_model(model, train, test)
    save_model(model)
