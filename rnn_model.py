
import torch
import torch.nn as nn
import numpy as np


class RNNCell(nn.Module):

    
    def __init__(self, inputSize, hiddenSize):
        super(RNNCell, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        self.Winputhead = nn.Parameter(torch.randn(hiddenSize, inputSize) * 0.01)
        self.binputhead = nn.Parameter(torch.zeros(hiddenSize))
        
        self.Whidhead = nn.Parameter(torch.randn(hiddenSize, hiddenSize) * 0.01)
        self.bhidhead = nn.Parameter(torch.zeros(hiddenSize))
        
    def forward(self, x, hidden):
        
        return  torch.tanh(torch.matmul(x, self.Winputhead.t()) +self.binputhead +torch.matmul(hidden, self.Whidhead.t()) + self.bhidhead)


class RNNfromScratch(nn.Module):
    def __init__(self,inputSize = 1,hiddenSize = 64,outputSize = 5,numLayers = 2,dropout = 0.2):
        super(RNNfromScratch, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.dropoutchance = dropout
        for i in range(numLayers):
            if i == 0:
                self.rnns = nn.ModuleList([RNNCell(inputSize)])
            else:
                self.rnns = nn.ModuleList([RNNCell(hiddenSize, hiddenSize)])
        
        if dropout > 0:
            self.dropout =nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.fullyconnectout = nn.Linear(hiddenSize, outputSize)
    
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

                hidden = self.rnns[index](inputT, hidden)
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
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output, y = self.forward(x)
        return output


class RNNForecaster:
    def __init__(self,inputSize = 1,hiddenSize =64,outputSize = 5,numLayers = 2, dropout = 0.2,learning_rate = 0.001,device =None):
        

        self.device = torch.device(device)
        
        self.model = RNNfromScratch(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=outputSize,numLayers=numLayers,dropout=dropout).to(self.device)
        
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
    
