## FutureCrypto

Predict the future `BTCUSDT` cryptocurrency exchange prices using neural network.

## Usage

1. Install the dependences:

   ```sh
   $ pip install -U numpy tensorflow matplotlib
   ```

2. Build the `BTCUSDT` dataset using [CryptocurrencyPriceDataset](https://github.com/NKID00/CryptocurrencyPriceDataset).

3. Create a symbolic directory link named `data` targeting the `data` directory containing the datasets.

4. Preprocess the data:

   ```sh
   $ python ./preprocess.py
   ```

   Preprocessed data is saved in `./preprocessed/BTCUSDT.npz`.


5. Train the model:

   ```sh
   $ python ./train.py
   ```

   Trained model is saved on every epoch in `./model/<Time>_<Sum of previous epoches>_<Current epoch>/`.

   Restore semi-trained model from the latest epoch:

   ```sh
   $ python ./restore.py
   ```

6. Plot the prediction (and real values):

   ```sh
   $ python ./plot.py
   ```

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