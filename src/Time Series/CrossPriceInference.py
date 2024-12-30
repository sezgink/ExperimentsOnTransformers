from crossPriceTransformer import CrossPriceTransformer
import sys
import os
import torch
from CollectRTMetaData import PrepeareRTData
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
##Model implementation
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

metas=["binance_btcusdt","binance_ethusdt","binance_solusdt"]

# Initialize the model
# input_dim = 16*5
# input_dim = 84 #dynamic input dim
input_dim = 84 #dynamic input dim
print("Input dim is: ",input_dim)
# target_dim = len(metas) * 4  # Number of target features
target_dim = len(metas) * 3  # Number of target features
# future_len = 2
future_len = 1
# input_dim = X_train.shape[2]  # Number of input features
# target_dim = Y_train.shape[2]  # Number of target features

model = CrossPriceTransformer(
    input_dim=input_dim, 
    d_model=512, 
    n_heads=16, 
    num_encoder_layers=5, 
    dim_feedforward=2048,
    dropout=0.1,
    future_seq_len=future_len, 
    target_dim=target_dim
).to(device)

existing_cp = None
if len(sys.argv)>1:
    existing_cp = sys.argv[1]

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float('inf')
recent_epoch = 0
##Activate for load
if existing_cp is not None:
    checkpoint = torch.load(existing_cp, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    recent_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']

is_calculating=True

model.eval()

# Initialize scaler
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler_adr = "scaler.pkl"
if len(sys.argv)>2:
    scaler_adr = sys.argv[2]

# to load the scaler
# scaler : StandardScaler = joblib.load(scaler_adr)
scaler : MinMaxScaler = joblib.load(scaler_adr)

last_pred = None
while is_calculating:
    rt_data = PrepeareRTData()
    if 'Time' in rt_data.columns:
        rt_data = rt_data.drop('Time', axis=1)
    if 'datetime' in rt_data.columns:
        rt_data = rt_data.drop('datetime', axis=1)
    daily_rsi =rt_data['binance_btcusdt_RSI_DAILY'].to_numpy()
    hourly_rsi  =rt_data['binance_btcusdt_RSI_HOURLY'].to_numpy()
    minute_cos  =rt_data['minute_cos'].to_numpy()
    minute_sin  =rt_data['minute_sin'].to_numpy()
    rt_data = rt_data.drop(['binance_btcusdt_RSI_DAILY','binance_btcusdt_RSI_HOURLY','minute_cos','minute_sin'],axis=1)
    print(rt_data.columns.to_list())
    feature_columns = rt_data.columns.tolist()
    features = rt_data[feature_columns]
    scaled = scaler.transform(features)
    
    # scaled['minute_cos'] = minute_cos
    # scaled['minute_sin'] = minute_sin
    # scaled['binance_btcusdt_RSI_DAILY'] = daily_rsi
    # scaled['binance_btcusdt_RSI_HOURLY'] = hourly_rsi

    scaled = np.column_stack((scaled, minute_cos, minute_sin,hourly_rsi ,daily_rsi))


    rt_data = scaled
    
    # rt_data.drop('datetime')
    # print(rt_data.head(5))
    # print(rt_data.count)
    # print(rt_data.keys)
    # rt_data.drop('Time')
   
    # rt_data = rt_data.apply(pd.to_numeric, errors='coerce')
    # rt_data = rt_data.fillna(0) 
    # rt_data = rt_data.values
    rt_data = torch.tensor(rt_data,dtype=torch.float32)
    rt_data = rt_data.unsqueeze(0)
    print(rt_data)
    rt_data = rt_data.to(device)
    pred = model(rt_data)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    print(pred)
    columns2reverse = [1,2,3]

    scaler_min = scaler.data_min_[columns2reverse]
    scaler_max = scaler.data_max_[columns2reverse]
    
    # pred_subset = pred[[0,1,2]]
    # print(f"Shape of pred: {pred.shape}")
    # pred_subset = pred[0, :3]

    pred_subset = pred[0, 0, :3]  # Shape will be (3,)
    print(f"Shape of pred_subset: {pred_subset.shape}")

    # Perform the scaling reversal
    reversed_subset = (pred_subset * (scaler_max-scaler_min)) + scaler_min
    print(f"Reversed subset: {reversed_subset}")
    if last_pred is not None:
        print(last_pred)
    last_pred = reversed_subset

    # time.sleep(60)
    time.sleep(10)
