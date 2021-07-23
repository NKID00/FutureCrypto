import tensorflow as tf
import numpy as np

from config import DATA_SIZE, SUBSAMPLING_SIZE, INPUT_SIZE, SHIFT_SIZE


def preprocess_data():
    print('load data ...')
    arr = np.load('./data/BTCUSDT.npz')['arr_0'][-DATA_SIZE:]

    print('preprocess data ...')
    xp = (arr.min(), arr.max())
    # subsample and scale to [-1.0, +1.0]
    arr = np.interp(arr[::SUBSAMPLING_SIZE], xp, [-1.0, +1.0])
    # split into chunks of INPUT_SIZE
    raw = arr[:len(arr) // INPUT_SIZE * INPUT_SIZE]
    raw = raw.reshape(len(raw) // INPUT_SIZE, INPUT_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices((arr,)).window(
        INPUT_SIZE, shift=SHIFT_SIZE, drop_remainder=True
    )
    arr = np.empty((len(dataset), INPUT_SIZE), dtype=np.float64)
    for i, ds in enumerate(dataset):
        print(f'preprocess row {i}/{arr.shape[0]} ...', end='\r')
        arr[i] = list(ds[0].as_numpy_iterator())
    print()

    # after 1 input_size
    y = arr[1:]
    arr = arr[:-1]
    arr = arr.reshape(arr.shape[0], arr.shape[1], 1)

    print('save data ...')
    np.savez_compressed(
        './preprocessed/BTCUSDT.npz',
        arr=arr, y=y, xp=xp, raw=raw
    )


if __name__ == '__main__':
    preprocess_data()
