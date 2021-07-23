## FutureCrypto

Predict the future `BTCUSDT` cryptocurrency exchange prices using neural network.

## Build

1. Build the `BTCUSDT` dataset using [CryptocurrencyPriceDataset](https://github.com/NKID00/CryptocurrencyPriceDataset).

2. Create a symbolic directory link named `data` targeting the `data` directory containing the datasets.

```sh
$ python ./preprocess_data.py
$ python ./train.py
$ python ./plot.py
```

Preprocessed data is saved in `./preprocessed/BTCUSDT.npz`, trained model is saved in `./model/<Time>/`.

## Model Structure

```
┌────────────┐
│ InputLayer │
└─────┬──────┘
      │
┌─────▼─────┐
│    GRU    │
└─────┬─────┘
      │
┌─────▼─────┐
│   LTSM    │
└─────┬─────┘
      │
┌─────▼─────┐
│   Dense   │
└─────┬─────┘
      │
┌─────▼─────┐
│   Dense   │
└───────────┘
```