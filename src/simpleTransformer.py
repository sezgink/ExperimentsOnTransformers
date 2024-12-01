import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size, sequnce_length, d_model, h_count, dropout=None) -> None: 
        super().__init__()

        #Input shape(batch_size,sequence_length,d_model)

        self.d_model = d_model
        self.h_count = h_count
        assert d_model % h_count == 0
        self.d_head = d_model // h_count

        #Key, Query and Value matrices
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model) 
        self.W_v = nn.Linear(d_model,d_model)

        #Output transformation matrix
        self.W_o = nn.Linear(d_model,d_model)

        # Optional dropout layer
        self.dropout = nn.Dropout(dropout) if dropout else None
    
    #method to seperate tensor
    def separate_tensor(self,X):
        print(X.size()) #(batch_size,seq_length,d_model)
        batch_size, seq_length, _ = X.size()
        return X.view(batch_size,seq_length,self.h_count,self.d_head).transpose(1,2)
    #method to concat tensors
    def concat_tensor(self,X):
        print(X.size())
        batch_size, _,seq_length, _ = X.size()
        return X.transpose(1,2).contiguous().view( batch_size,seq_length,self.d_model)
    #method for apply mask for 0 values in the mask
    @staticmethod
    def apply_mask(X,mask):
        return X.masked_fill(mask==0,-1e9)

    def forward(self, q, k ,v, mask=None):
        Query = self.W_q(q)
        Key = self.W_k(k)
        Value = self.W_v(v)

        Query= self.separate_tensor(Query)
        Key= self.separate_tensor(Key).transpose(-2,-1)
        Value= self.separate_tensor(Value)

        output = torch.matmul(Query,Key) #Score 
        output = output / math.sqrt(self.d_head) #Scale factor
        if mask is not None:
            output=self.apply_mask(output,mask)

        output=torch.softmax(output,-1) #Attention

        if self.dropout:
            output = self.dropout(output) #Dropout if dropout defined for layer

        output = torch.matmul(output,Value)
        output = self.concat_tensor(output)
        output = self.W_o(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=3000) -> None:
        super().__init__()

        position = torch.arange(0,max_length).unsqueeze(1)
        div_term = torch.arange(0,d_model,2) * (-torch.log(torch.tensor(10000.0)) / d_model)

        pe = torch.zeros(max_length,d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as buffer to avoid tracking gradients

    def forward(self,x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

        
        





