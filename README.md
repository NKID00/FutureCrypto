## FutureCrypto

Predict the future `BTCUSDT` cryptocurrency exchange prices using neural network.

## Usage

1. Install the dependences:

   ```sh
   $ pip install -U numpy tensorflow matplotlib
   ```

2. Build the `BTCUSDT` dataset using [CryptocurrencyPriceDataset](https://github.com/NKID00/CryptocurrencyPriceDataset).

3. Create a symbolic directory link named `data` targeting the `data` directory containing the datasets.

4. Modify `./config.py` if needed.

5. Run this script to preprocess the data:

   ```sh
   $ python ./preprocess.py
   ```

   Preprocessed data is saved in `./preprocessed/BTCUSDT.npz`.


6. Run this script to train the model:

   ```sh
   $ python ./train.py
   ```

   Trained model is saved on every epoch in `./model/<Time>_<Sum of previous epoches>_<Current epoch>/`.

   Or run this script to restore semi-trained model from the latest epoch:

   ```sh
   $ python ./restore.py
   ```

7. Run this script to plot the prediction (and real values):

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