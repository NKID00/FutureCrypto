## FutureBTCUSDT

Predict the future `BTCUSDT` prices using neural network.

## Build

1. Build `BTCUSDT`, `ETHUSDT` and `ETHBTC` dataset using [CryptocurrencyPriceDataset](https://github.com/NKID00/CryptocurrencyPriceDataset).

2. Create a symbolic directory link named `data` targeting the `data` directory containing the datasets.

```sh
$ python ./main.py
```

Trained model is saved in `./model/<Time>/`.

## Model Structure

```
    ┌──────────────────────┐ ┌──────────────────────┐ ┌─────────────────────┐
    │ btcusdt (InputLayer) │ │ ethusdt (InputLayer) │ │ ethbtc (InputLayer) │
    └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬──────────┘
               │                        │                        │
       ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
       │ lstm_1 (LSTM) │        │ lstm_2 (LSTM) │        │ lstm_3 (LSTM) │
       └─┬───────────┬─┘        └───────┬───────┘        └─┬─────────────┘
         │           └────────┐         │          ┌───────┘
┌────────▼─────────┐        ┌─▼─────────▼──────────▼─┐
│ output_1 (Dense) │        │ concat_1 (Concatenate) │
└──────────────────┘        └───────────┬────────────┘
                                        │
                               ┌────────▼────────┐
                               │ dense_1 (Dense) │
                               └────────┬────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  output_2 (Dense)  │
                              └────────────────────┘
```