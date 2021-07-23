from typing import Tuple
from datetime import datetime

import tensorflow as tf
import numpy as np

from config import (
    INPUT_SIZE, OUTPUT_SIZE, TEST_DATASET_SIZE,
    SHUFFLE_BUFFER_SIZE, BATCH_SIZE, EPOCHS
)
from train import load_train_data
from plot import load_model

def restore_train_model(
    model: tf.keras.Model, name:str,
    train: tf.data.Dataset, test: tf.data.Dataset
):
    if '_' not in name:
        print('no semi-trained model found!')
        return
    name, start_from, epochs_trained = name.split('_')
    start_from = int(start_from)
    epochs_trained = int(epochs_trained)
    print('train model from %d epoch ...' % (
        start_from + epochs_trained
    ))
    model.fit(train, epochs=(
        EPOCHS - start_from - epochs_trained
    ), verbose=1, callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './model/%s_%02d_{epoch:02d}' % (name, start_from + epochs_trained)
        )
    ])
    model.evaluate(test, verbose=1)


if __name__ == '__main__':
    model, name = load_model()
    train, test = load_train_data()
    restore_train_model(model, name, train, test)