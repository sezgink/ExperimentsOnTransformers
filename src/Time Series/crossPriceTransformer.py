import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len]
        return x


# Transformer-based Model
class CrossPriceTransformer(nn.Module):
    def __init__(self, 
                 input_dim,       # number of features in each time step
                 d_model=128,     # embedding dimension for transformer
                 n_heads=8,       # number of attention heads
                 num_encoder_layers=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 future_seq_len=5,
                 target_dim=30,   # dimension of target per output step (e.g., predicting close price for 30 cryptos)
                 max_len=5000):
        super(CrossPriceTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.future_seq_len = future_seq_len
        
        # Linear projection from input features to d_model with Linear Embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=n_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_encoder_layers)
        
        # Decoder for forecasting future steps:
        # We will treat this as generating next steps in parallel.
        # A simple approach: 
        # After encoding the historical sequence, we use the final hidden state or 
        # a pooled representation to produce the next future_seq_len steps.
        
        # One approach: Take the last hidden vector of the sequence as a representation
        # and then map it to the next future steps. 
        # Alternatively, we can feed a "future token" sequence into a Transformer decoder. 
        # For simplicity, let's do a simpler approach:
        # We'll average pool over the time dimension and use that as a summary.
        
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool over seq dimension
        self.fc_out = nn.Linear(d_model, future_seq_len * target_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. Project input features to d_model dimension
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 2. Add positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        
        # 3. Pass through transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 4. Pool the sequence to get a single representation
        # encoded: (batch_size, seq_len, d_model)
        # We can transpose to (batch_size, d_model, seq_len) for pooling
        encoded_t = encoded.transpose(1, 2)  # (batch_size, d_model, seq_len)
        pooled = self.pooling(encoded_t).squeeze(-1)  # (batch_size, d_model)
        
        # 5. Generate future predictions
        # output shape: (batch_size, future_seq_len * target_dim)
        out = self.fc_out(pooled)
        
        # reshape to (batch_size, future_seq_len, target_dim)
        out = out.view(-1, self.future_seq_len, self.target_dim)
        return out


# Dummy Dataset Class
class DummyCrossPriceDataset(Dataset):
    def __init__(self, data, target, seq_len=60, future_len=5):
        # data: tensor of shape (num_samples, seq_len, input_dim)
        # target: tensor of shape (num_samples, future_len, target_dim)
        self.data = data
        self.target = target
        self.seq_len = seq_len
        self.future_len = future_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx]    # (seq_len, input_dim)
        Y = self.target[idx]  # (future_len, target_dim)
        return X, Y


# Training Loop Example
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    # Example usage
    print("Cuda availability:",torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppose we have 30 intruments, each with Open,High,Low,Close,Volume = 5 features => 30*5=150 features
    # Add RSI, MA, Fed rate, etc. Let's say total input_dim = 200 for example
    input_dim = 200
    target_dim = 30  # Predicting one value per instrument, for instance close price
    
    seq_len = 60   # 60 past timesteps (5-minute candles)
    future_len = 5 # predict next 5 steps
    
    # Dummy data for demonstration (10,000 samples)
    num_samples = 10000
    data = torch.randn(num_samples, seq_len, input_dim)
    target = torch.randn(num_samples, future_len, target_dim)
    
    dataset = DummyCrossPriceDataset(data, target, seq_len=seq_len, future_len=future_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train for a few epochs
    for epoch in range(5):
        train_loss = train_model(model, dataloader, optimizer, criterion, device)
        val_loss = validate_model(model, dataloader, criterion, device)  # In real scenario, separate train & val
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
