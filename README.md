## FutureBTCUSDT

Predict the future `BTCUSDT` prices using neural network.

## Build

1. Build `BTCUSDT`, `ETHUSDT` and `ETHBTC` dataset using [CryptocurrencyPriceDataset](https://github.com/NKID00/CryptocurrencyPriceDataset).

2. Create a symbolic directory link named `data` targeting the `data` directory containing the datasets.

3. ```sh
   $ python ./main.py
   ```

Trained model is saved in `./model/<Time>/`.

## Model Structure (TODO)

```
    ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
    │ input_1 (InputLayer) │ │ input_2 (InputLayer) │ │ input_3 (InputLayer) │
    │  (BTCUSDT dataset)   │ │  (ETHUSDT dataset)   │ │   (ETHBTC dataset)   │
    └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
               │                        │                        │
       ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
       │ lstm_1 (LSTM) │        │ lstm_2 (LSTM) │        │ lstm_3 (LSTM) │
       └─┬───────────┬─┘        └───────┬───────┘        └─┬─────────────┘
         │           └───────────┐      │      ┌───────────┘
┌────────▼─────────┐           ┌─▼──────▼──────▼─┐
│ output_1 (Dense) │           │ merge_1 (Merge) │
│  (Main output)   │           └────────┬────────┘
└──────────────────┘                    │
                               ┌────────▼────────┐
                               │ dense_1 (Dense) │
                               └────────┬────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  output_2 (Dense)  │
                              │ (Auxiliary output) │
                              └────────────────────┘
```