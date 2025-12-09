import torch
import torch.nn as nn
import numpy as np
import math


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout = 0.1, context_size = None):
        super(GatedResidualNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate_fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None
        
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context):
        n2 = self.elu(self.fc1(x))
        
        if context is not None and self.context_size is not None:
            n2 = n2 + self.context_fc(context)
        
        n2 = self.dropout(self.fc2(n2))
        n1 = self.gate_fc(n2)
        gate = self.sigmoid(n1)
        
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
        
        output = self.layer_norm(skip + gate * n1)
        
        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        self.grn_weights = GatedResidualNetwork(
            input_size=input_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout
        )
        
        self.grn_vars = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
            for z in range(num_inputs)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size = x.size(0)
        flatten = x.view(batch_size, -1)
        weights = self.grn_weights(flatten)
        weights = self.softmax(weights)
        var_outputs = []
        for i, grn in enumerate(self.grn_vars):
            var_outputs.append(grn(x[:, i, :]))
        
        var_outputs = torch.stack(var_outputs, dim=1)
        
        weights_expanded = weights.unsqueeze(-1) 
        output = torch.sum(var_outputs * weights_expanded, dim=1) 
        
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, mask):
        batch_size = queries.size(0)
        Q = self.W_q(queries).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attn_output = torch.matmul(attention_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 input_size = 1,
                 hidden_size = 64,
                 num_heads = 4,
                 num_encoder_layers = 2,
                 output_size = 5,
                 dropout = 0.1):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size
        
        self.input_embedding = nn.Linear(input_size, hidden_size)
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            num_inputs=1,  
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0
        )
        self.post_lstm_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.post_attention_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.final_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_len, z = x.shape
        x_reshaped = x.view(batch_size * seq_len, 1, self.input_size)
        selected, var_weights = self.vsn(x_reshaped)
        selected = selected.view(batch_size, seq_len, self.hidden_size)
        lstm_out, z = self.lstm_encoder(selected)
        lstm_out = self.post_lstm_grn(lstm_out)
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.post_attention_grn(attn_out)
        combined = self.layer_norm(lstm_out + attn_out)
        ff_out = self.feed_forward(combined)
        combined = self.layer_norm(combined + ff_out)
        final = self.final_grn(combined)
        pooled = torch.mean(final, dim=1)
        output = self.output_projection(pooled)
        interpretations = {
            'variable_selection_weights': var_weights,
            'attention_weights': attention_weights
        }
        
        return output, interpretations
    
    def predict(self, x) :
        self.eval()
        with torch.no_grad():
            output, z = self.forward(x)
        return output.cpu().numpy()


class TFTForecaster:
    def __init__(self,
                 input_size = 1,
                 hidden_size = 64,
                 num_heads = 4,
                 num_encoder_layers = 2,
                 output_size = 5,
                 dropout = 0.1,
                 learning_rate = 0.001,
                 device = None):
        

        self.device = torch.device(device)
        
        self.model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
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
                output, z = self.model(X_batch)
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
                    val_output, z = self.model(X_val_tensor)
                    val_loss = self.criterion(val_output, y_val_tensor).item()
                    self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                if has_validation:
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if has_validation else None
        }
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        return self.model.predict(X_tensor)
    