import tensorflow as tf

from config import EPOCHS
from train import load_train_data
from plot import load_model

def restore_train_model(
    model: tf.keras.Model, name:str,
    train: tf.data.Dataset
):
    if '_' not in name:
        print('no semi-trained model found!')
        return
    name, start_from, epochs_trained = name.split('_')
    start_from = int(start_from)
    epochs_trained = int(epochs_trained)
    if start_from + epochs_trained >= EPOCHS:
        print('no semi-trained model found!')
    print('train model from %d epoch ...' % (
        start_from + epochs_trained
    ))
    model.fit(train, epochs=(
        EPOCHS - start_from - epochs_trained
    ), verbose=1, callbacks=[
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='loss', patience=2, restore_best_weights=True
        # ),
        tf.keras.callbacks.ModelCheckpoint(
            './model/%s_%02d_{epoch:02d}' % (name, start_from + epochs_trained)
        )
    ])


if __name__ == '__main__':
    model, name = load_model()
    train = load_train_data()
    restore_train_model(model, name, train)
