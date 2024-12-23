from crossPriceTransformer import CrossPriceTransformer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from MultiMetricPriceDataset import MultiMetricPriceDataset
import numpy as np
import os
import datetime
##Data importing
print("Step1")

dataAdress = "../../topsecret/cumulative_data.parquet" #default data adress if data adress not given as command line argument
if len(sys.argv)>1:
    dataAdress = sys.argv[1]
existing_cp = None
if len(sys.argv)>2:
    existing_cp = sys.argv[2]

df = pd.read_parquet(dataAdress)

print(df.head(5))

# Convert 'datetime' column to datetime objects
df['datetime'] = pd.to_datetime(df['datetime'])

# Set 'datetime' as the index
df.set_index('datetime', inplace=True)

# drop the 'timestamp' column
df.drop(columns=['timestamp'], inplace=True)

print(df.head())

print("Step2")

# Check for missing values
print(df.isnull().sum())

# Handle missing values if any (with forward fill)
df.fillna(method='ffill', inplace=True)

# Select features (all columns as inputs, including RSI)
feature_columns = df.columns.tolist()

# Example: ['binance_btcusdt_Open', 'binance_btcusdt_High', ..., 'binance_btcusdt_RSI_1d', ...]
features = df[feature_columns]

# Initialize scaler
scaler = StandardScaler()

# to load the scaler
# scaler = joblib.load('scaler.pkl')

# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Save the scaler to a file
joblib.dump(scaler, 'checkpoints/scaler.pkl')

# Convert back to DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_features, index=df.index, columns=feature_columns)

# Quick checks before creating the dataset
print("Checking if any NaN in dataframe:", df.isna().any().any())
print("Checking if any Inf in dataframe:", np.isinf(df.values).any())

# Also check your scaled data:
print("Checking if any NaN in scaled_df:", scaled_df.isna().any().any())
print("Checking if any Inf in scaled_df:", np.isinf(scaled_df.values).any())


print("Step3")

## Implement dataset and dataloader

# Parameters
SEQ_LENGTH = 40  # Past 60 minutes
TARGET_COLUMN = 'binance_btcusdt_Close'  # Example target

# Initialize Dataset
# dataset = MultiMetricPriceDataset(scaled_df, seq_length=SEQ_LENGTH, target_column=TARGET_COLUMN)
# metas=["binance_btcusdt","binance_ethusdt","binance_solusdt","coinbase_btcusd"]
metas=["binance_btcusdt","binance_ethusdt","binance_solusdt"]
dataset = MultiMetricPriceDataset(scaled_df, seq_length=SEQ_LENGTH, future_seq_len=2, metas=metas )

sample_data = dataset[1]
print("Sample Data 1")
print(sample_data)

print("Shape",dataset[0][1].shape)

print("Last sample",dataset[-1])

print("Step4")

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

sample_data = train_dataset[1]
print("Sample Data 2")
print(sample_data)



# Create DataLoaders
BATCH_SIZE = 64

print("Step5")

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #Don't shuffle yet
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Step6")

##Model implementation
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
input_dim = 16*5
# target_dim = len(metas) * 4  # Number of target features
target_dim = len(metas) * 3  # Number of target features
# future_len = 2
future_len = 1
# input_dim = X_train.shape[2]  # Number of input features
# target_dim = Y_train.shape[2]  # Number of target features

print("Train dataset",train_dataset[0])
print("Target dim is",target_dim)

model = CrossPriceTransformer(
    input_dim=input_dim, 
    d_model=128, 
    n_heads=8, 
    num_encoder_layers=4, 
    dim_feedforward=256,
    dropout=0.1,
    future_seq_len=future_len, 
    target_dim=target_dim
).to(device)


# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training and Validation Functions
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # for X_batch, Y_batch in dataloader:
    for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)

        # # 1. Check if predictions are already NaN
        # if torch.isnan(preds).any():
        #     print(f"[Batch {batch_idx}] preds contain NaN.")
        #     break

        # # 2. Check shapes
        # if preds.shape != Y_batch.shape:
        #     print(f"[Batch {batch_idx}] shape mismatch: preds={preds.shape}, Y_batch={Y_batch.shape}")
        #     break

        loss = criterion(preds, Y_batch)

        # if torch.isnan(loss):
        #     print(f"[Batch {batch_idx}] loss is NaN.")
        #     break

        loss.backward()

        # 4. Check gradients for NaNs
        # for name, param in model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"[Batch {batch_idx}] NaN in gradients at layer {name}.")
        #         break
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

# Training Loop
num_epochs = 20  # Adjust based on your requirements

print("Before training")

## Activate for anomaly detection
# torch.autograd.set_detect_anomaly(True)
##

# #Test evaoluation for check problems
# X_batch, Y_batch = next(iter(train_loader))
# X_batch = X_batch.to(device)
# Y_batch = Y_batch.to(device)

# with torch.no_grad():
#     preds = model(X_batch)
#     print("Preds:", preds[:1])  # Check the first item of the batch
#     print("Targets:", Y_batch[:1])

# # Also check whether the loss is NaN even before backward:
# loss = criterion(preds, Y_batch)
# print("Initial loss:", loss.item())

# print("Before training 2")

# Directory to save checkpoints
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float('inf')
best_model_path = os.path.join(checkpoint_dir, f'best_model{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')

##Activate for load
if existing_cp is not None:
    checkpoint = torch.load(existing_cp, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    recent_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['val_loss']

## Start training

# for epoch in range(num_epochs):
for epoch in range(recent_epoch,num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss = validate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save the model and optimizer state_dict if its validation loss is better
    if val_loss>best_val_loss:
        continue
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, best_model_path)
    print(f"Checkpoint saved at {best_model_path}")
    best_val_loss = val_loss
