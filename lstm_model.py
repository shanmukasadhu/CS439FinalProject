import torch
import torch.nn as nn
import numpy as np


class LSTMCell(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(LSTMCell, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        self.Wforgot = nn.Parameter(torch.randn(hiddenSize, inputSize + hiddenSize) *0.01)
        self.bforgot = nn.Parameter(torch.zeros(hiddenSize))
        
        self.winitial = nn.Parameter(torch.randn(hiddenSize, inputSize + hiddenSize) *0.01)
        self.binitial = nn.Parameter(torch.zeros(hiddenSize))
        
        self.Wgate = nn.Parameter(torch.randn(hiddenSize, inputSize + hiddenSize) *0.01)
        self.bgate = nn.Parameter(torch.zeros(hiddenSize))
        
        self.Wouput = nn.Parameter(torch.randn(hiddenSize, inputSize + hiddenSize) * 0.01)
        self.bouput = nn.Parameter(torch.zeros(hiddenSize))
        
    def forward(self, x,hidden):
        h = hidden
        c = hidden
        
        combined = torch.cat([h, x], dim=1)
        
        forgot = torch.sigmoid(torch.matmul(combined, self.Wforgot.t()) +self.bforgot)
        
        inital = torch.sigmoid(torch.matmul(combined, self.winitial.t())+ self.binitial)
        
        gate = torch.tanh(torch.matmul(combined, self.Wgate.t()) +self.bgate)
        output = torch.sigmoid(torch.matmul(combined, self.Wouput.t())+ self.bouput)
        
        new = forgot *c + inital *gate
        new_h = output * torch.tanh(new)
        
        return new_h, new


class LSTMfromScratch(nn.Module):
    def __init__(self, inputSize = 1,hiddenSize = 64,outputSize = 5,numLayers = 2,dropout= 0.2):
        super(LSTMfromScratch, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.dropoutchance = dropout
        
        for i in range(numLayers):
            if i == 0:
                self.lstms = nn.ModuleList([LSTMCell(inputSize)])
            else:
                self.lstms = nn.ModuleList([LSTMCell(hiddenSize, hiddenSize)])
        
        if dropout > 0:
            self.dropout =nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.fc_out = nn.Linear(hiddenSize, hiddenSize)
        
    
    def forward(self, x, hidden):
        bs = x.size(0)
        
        fhs = []

        for index in range(self.numLayers):
            hidden = hidden[index]
            outputs = []

            for t in range(x.size(1)):
                if index == 0:
                    inputT = x[:,t, :]
                else:
                    inputT = outputs_prev[:, t, :]

                hidden = self.lstms[index](inputT, hidden)
                outputs.append(hidden)

            outputs = torch.stack(outputs, dim=1)

            if self.dropout is not None:
                if index < self.numLayers - 1:
                    outputs = self.dropout(outputs)

            fhs.append(hidden)

            outputs_prev = outputs

        hidden = torch.stack(fhs, dim=0)
        last_hidden = outputs[:, -1, :]
        output = self.fullyconnectout(last_hidden) 
        
        return output, hidden
    
    def _init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hiddenSize).to(device)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
        return output.cpu().numpy()


class LSTMForecaster:
    def __init__(self,inputSize = 1,hiddenSize =64,outputSize = 5,numLayers = 2, dropout = 0.2,learning_rate = 0.001,device =None):
        

        self.device = torch.device(device)
        
        self.model = LSTMfromScratch(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=outputSize,numLayers=numLayers,dropout=dropout).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.trainLosses = []
        
        self.valLosses = []
    
    def fit(self,X_train, y_train,X_val,y_val,epochs= 100,bs =32,verbose = True):
        Xtrain = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        ytrain = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            Xval = torch.FloatTensor(X_val).unsqueeze(-1).to(self.device)
            yval = torch.FloatTensor(y_val).to(self.device)
            validation = True
        else:
            validation = False
        
        for epoch in range(epochs):
            self.model.train()
            epochLoss = 0
            nBatches = 0
            
            indices = torch.randperm(len(Xtrain))
            
            for i in range(0, len(Xtrain), bs):
                batch_indices = indices[i:i+bs]
                X_batch = Xtrain[batch_indices]
                y_batch = ytrain[batch_indices]
                
                self.optimizer.zero_grad()
                output, _ = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epochLoss += loss.item()
                nBatches += 1
            
            avgTrainLoss = epochLoss / nBatches
            self.trainLosses.append(avgTrainLoss)
            
            if validation:
                self.model.eval()
                with torch.no_grad():
                    valOutput,z = self.model(Xval)
                    
                    val_loss = self.criterion(valOutput, yval).item()
                    self.valLosses.append(val_loss)
                    

        
        return {'train_loss': self.trainLosses,'val_loss': self.valLosses}
    
    def predict(self, X):
        return self.model.predict(torch.FloatTensor(X).unsqueeze(-1).to(self.device))
