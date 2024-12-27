from crossPriceTransformer import CrossPriceTransformer

##Model implementation
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

metas=["binance_btcusdt","binance_ethusdt","binance_solusdt"]

# Initialize the model
# input_dim = 16*5
input_dim = inputShape[1] #dynamic input dim
print("Input dim is: ",input_dim)
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
    d_model=256, 
    n_heads=8, 
    num_encoder_layers=5, 
    dim_feedforward=512,
    dropout=0.1,
    future_seq_len=future_len, 
    target_dim=target_dim
).to(device)