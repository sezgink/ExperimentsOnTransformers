import torch
from torch.utils.data import Dataset
import numpy as np
from pandas import DataFrame
class MultiMetricPriceDataset(Dataset):

    def __init__(self, data : DataFrame, seq_length : int, future_seq_len : int, metas,calculate_rsi=False,add_daytime_encoding=False):
        """
        Args:
            data (DataFrame): The preprocessed and scaled data.
            seq_length (int): Number of past time steps to include in each input sequence.
            future_seq_len (int): Number of future time steps to predict.
            metas (list): List of meta identifiers.
        """
        self.seq_length = seq_length
        self.future_seq_len = future_seq_len
        self.data = data
        self.metas = metas
        self.num_metas = len(metas)

        
        # Precompute target indices for efficiency
        self.target_columns = []
        for meta in metas:
            self.target_columns.extend([
                f'{meta}_High',   # For max high
                f'{meta}_Low',    # For min low
                f'{meta}_Close'   # For close at future_seq_len
            ])
        self.target_idxs = [data.columns.get_loc(col) for col in self.target_columns]
        
        # Precompute the targets
        self.targets = self._compute_targets_vectorized()

        if add_daytime_encoding:
            self._add_daytime_encoding()

        if calculate_rsi:
            metas2calculate = ["binance_btcusdt"]
            self._calculate_RSI(metas2calculate)

    @staticmethod
    def RSI_From_Data(df : DataFrame,window_size=14):
        # Step 3: Separate positive and negative changes
        diffs = df.diff()
        gains = diffs.clip(lower=0)  # Positive changes
        losses = -diffs.clip(upper=0)  # Negative changes (inverted sign)

        avg_gain = gains.rolling(window=window_size, min_periods=window_size).mean()
        avg_loss = losses.rolling(window=window_size, min_periods=window_size).mean()

        #Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
        
    def _calculate_RSI(self,metas2calculate):
        hourly_resampled = self.data.resample('H').last()
        daily_resampled = self.data.resample('D').last()
        for meta in metas2calculate:
            self.data[f'{meta}_RSI_HOURLY'] = self.RSI_From_Data(hourly_resampled[f'{meta}_Close'])
            self.data[f'{meta}_RSI_DAILY'] = self.RSI_From_Data(daily_resampled[f'{meta}_Close'])
        valid_indices = self.data[f'{metas2calculate[0]}_RSI_DAILY'].dropna().index
        self.data = self.data.loc[valid_indices]

    def _add_daytime_encoding(self):

        hour = self.data.index.hour
        minute = self.data.index.minute + hour*60

        # Normalize the minutes to the range [0, 2π]
        normalized_minutes = (minute / (24 * 60)) * 2 * np.pi
        
        # Add sine and cosine of the normalized time as new columns
        self.data['minute_cos'] = np.cos(normalized_minutes)
        self.data['minute_sin'] = np.sin(normalized_minutes)
            
    
    def _compute_targets_vectorized(self):
        """
        Computes the targets for each sample in the dataset using vectorized operations.

        Returns:
            torch.Tensor: Array of targets with shape (num_samples, num_metas * 4)
        """
        num_samples = len(self.data) - self.seq_length - self.future_seq_len + 1

        # Pre-allocate an array for targets
        # targets = np.zeros((num_samples, self.num_metas * 4), dtype=np.float32)
        targets = np.zeros((num_samples, self.num_metas * 3), dtype=np.float32)

        # Loop over metas (this is minimal and unavoidable here)
        for i, meta in enumerate(self.metas):
            # Extract relevant columns for this meta
            print(f"Creating targets for {meta} and index is {i}")

            high_col = f"{meta}_High"
            low_col = f"{meta}_Low"
            close_col = f"{meta}_Close"

            # Extract the rolling windows for future sequences
            high_values = self.data[high_col].rolling(window=self.future_seq_len).max().shift(-self.future_seq_len + 1)
            low_values = self.data[low_col].rolling(window=self.future_seq_len).min().shift(-self.future_seq_len + 1)
            close_first_values = self.data[close_col].shift(-self.seq_length)
            # close_future_values = self.data[close_col].shift(-self.seq_length - self.future_seq_len + 1)


            # Limit to valid indices
            high_values = high_values.iloc[self.seq_length:self.seq_length + num_samples]
            low_values = low_values.iloc[self.seq_length:self.seq_length + num_samples]
            close_first_values = close_first_values.iloc[:num_samples]
            # close_future_values = close_future_values.iloc[:num_samples]

            # Add targets for this meta to the target array
            targets[:, i * 3] = high_values.values
            targets[:, i * 3 + 1] = low_values.values
            targets[:, i * 3 + 2] = close_first_values.values
            # # Add targets for this meta to the target array
            # targets[:, i * 4] = high_values.values
            # targets[:, i * 4 + 1] = low_values.values
            # targets[:, i * 4 + 2] = close_first_values.values
            # # targets[:, i * 4 + 3] = close_future_values.values

            
        print(targets[:5])
        return torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data) - self.seq_length - self.future_seq_len + 1
    
    def __getitem__(self, idx):
        # Input sequence: seq_length rows, all features
        x = self.data.iloc[idx : idx + self.seq_length].values
        x = torch.tensor(x, dtype=torch.float32)

        # if np.isnan(x).any() or np.isinf(x).any():
        #     print("NaN or Inf in X_batch at index", idx)
        
        # Target: precomputed targets
        y = self.targets[idx]
        # if torch.isnan(y).any() or torch.isinf(y).any():
        #     print("NaN or Inf in Y_batch at index", idx)    
        y = y.unsqueeze(0)


        # y = self.targets[idx].view(self.future_seq_len, -1)
        return x, y