import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len= 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask):
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask):
        batch_size = x.size(0)
        
        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,input_size= 1,d_model= 64,num_heads= 4,num_layers= 2,d_ff= 256,output_size= 5,dropout = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(d_model, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask):
        x = self.input_projection(x) 
        
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = torch.mean(x, dim=1)
        
        output = self.fc_out(x)
        
        return output
    
    def predict(self, x) :
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.cpu().numpy()


class TransformerForecaster:
    def __init__(self,
                 input_size= 1,
                 d_model= 64,
                 num_heads= 4,
                 num_layers= 2,
                 d_ff= 256,
                 output_size= 5,
                 dropout = 0.1,
                 learning_rate = 0.001,
                 device = None):
        

        self.device = torch.device(device)
        
        self.model = TransformerEncoder(
            input_size=input_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []

    
    def fit(self, 
            X_train, 
            y_train,
            X_val,
            y_val,
            epochs = 100,
            batch_size = 32,
            verbose = True):
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            has_validation = True
        else:
            has_validation = False
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            indices = torch.randperm(len(X_train_tensor))
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                self.optimizer.zero_grad()
                output, _ = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    val_output, _ = self.model(X_val_tensor)
                    val_loss = self.criterion(val_output, y_val_tensor).item()
                    self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if has_validation else None
        }
        
        return history
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        return self.model.predict(X_tensor)
    