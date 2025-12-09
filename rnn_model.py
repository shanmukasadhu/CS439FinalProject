
import torch
import torch.nn as nn
import numpy as np


class RNNCell(nn.Module):

    
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, hidden):
        
        return  torch.tanh(torch.matmul(x, self.W_ih.t()) + self.b_ih +torch.matmul(hidden, self.W_hh.t()) + self.b_hh)


class VanillaRNN(nn.Module):
    def __init__(self,input_size = 1,hidden_size = 64,output_size = 5,num_layers = 2,dropout = 0.2):
        super(VanillaRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        
        self.rnn_cells = nn.ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.rnn_cells:
            nn.init.xavier_uniform_(layer.W_ih)
            nn.init.xavier_uniform_(layer.W_hh)
            nn.init.zeros_(layer.b_ih)
            nn.init.zeros_(layer.b_hh)
        
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x, hidden):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        final_hidden_states = []

        for layer_idx in range(self.num_layers):
            layer_hidden = hidden[layer_idx]
            layer_outputs = []

            for t in range(seq_len):
                if layer_idx == 0:
                    input_t = x[:, t, :]
                else:
                    input_t = layer_outputs_prev[:, t, :]

                layer_hidden = self.rnn_cells[layer_idx](input_t, layer_hidden)
                layer_outputs.append(layer_hidden)

            layer_outputs = torch.stack(layer_outputs, dim=1)

            if self.dropout is not None and layer_idx < self.num_layers - 1:
                layer_outputs = self.dropout(layer_outputs)

            final_hidden_states.append(layer_hidden)

            layer_outputs_prev = layer_outputs

        hidden = torch.stack(final_hidden_states, dim=0)
        last_hidden = layer_outputs[:, -1, :]
        output = self.fc_out(last_hidden) 
        
        return output, hidden
    
    def _init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
        return output.cpu().numpy()


class RNNForecaster:
    def __init__(self,
                 input_size = 1,
                 hidden_size = 64,
                 output_size = 5,
                 num_layers = 2,
                 dropout = 0.2,
                 learning_rate = 0.001,
                 device: str = None):
        

        self.device = torch.device(device)
        
        self.model = VanillaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
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
            verbose: bool = True):
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
    
