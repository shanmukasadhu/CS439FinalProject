import torch
import torch.nn as nn
import numpy as np
import math


class GRN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, dropout = 0.1, contextSize = None):
        super(GRN, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.contextSize = contextSize
        
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.elu = nn.ELU()
        
        self.context_fc = nn.Linear(contextSize, hiddenSize, bias=False)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.gateFC = nn.Linear(hiddenSize, outputSize)
        self.sigmoid = nn.Sigmoid()
        
        self.skip_fc = nn.Linear(inputSize, outputSize)
        
        self.normaliz = nn.LayerNorm(outputSize)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context):
        n2 = self.elu(self.fc1(x))
        
        if context is not None:
            if self.contextSize is not None:
                n2 = n2 + self.context_fc(context)
        
        n2 = self.dropout(self.fc2(n2))
        n1 = self.gateFC(n2)
        gate = self.sigmoid(n1)
        
        skip = self.skip_fc(x)
        
        output = self.normaliz(skip +gate *n1)
        
        return output


class VSN(nn.Module):
    def __init__(self, inputSize, numInputs, hiddenSize, dropout = 0.1):
        super(VSN, self).__init__()
        
        self.inputSize = inputSize
        self.numInputs = numInputs
        self.hiddenSize = hiddenSize
        
        self.grnWeights = GRN(inputSize=inputSize *numInputs,hiddenSize=hiddenSize,outputSize=numInputs,dropout=dropout)
        
        self.vars = nn.ModuleList([GRN(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=hiddenSize,dropout=dropout)for z in range(numInputs)])
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        bs = x.size(0)
        flatten = x.view(bs, -1)
        weights = self.grnWeights(flatten)
        
        weights = self.softmax(weights)
        
        var_outputs = []
        for i, grn in enumerate(self.vars):
            
            var_outputs.append(grn(x[:, i, :]))
        
        var_outputs = torch.stack(var_outputs, dim=1)
        
        weights_expanded = weights.unsqueeze(-1) 
        
        output = torch.sum(var_outputs *weights_expanded, dim=1) 
        
        return output, weights


class MHA(nn.Module):
    def __init__(self, DModel, numHeads, dropout = 0.1):
        super(MHA, self).__init__()
        
        assert DModel %numHeads == 0
        
        self.DModel = DModel
        self.numHeads =numHeads
        self.keys =DModel //numHeads
        
        self.query = nn.Linear(DModel, DModel)
        self.key = nn.Linear(DModel, DModel)
        self.value = nn.Linear(DModel, DModel)
        self.ouput = nn.Linear(DModel, DModel)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys,values, mask):
        bs = queries.size(0)
        Q = self.query(queries).view(bs, -1, self.numHeads, self.keys).transpose(1, 2)
        
        K = self.key(keys).view(bs, -1, self.numHeads, self.keys).transpose(1, 2)
        
        V = self.value(values).view(bs, -1, self.numHeads, self.keys).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.keys)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = torch.softmax(scores, dim=-1)
        weights =self.dropout(weights)
        
        atoiptput= torch.matmul(weights, V)
        atoiptput = atoiptput.transpose(1, 2).contiguous().view(bs, -1, self.DModel)

        output =self.ouput(atoiptput)
        avg_attention= weights.mean(dim=1)
        
        return output, avg_attention


class TFT(nn.Module):
    def __init__(self,inputSize = 1,hiddenSize = 64,numHeads = 4,encodlays = 2,outputSize = 5,dropout = 0.1):
        super(TFT, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.outputSize = outputSize
        
        self.input_embedding = nn.Linear(inputSize, hiddenSize)
        self.vsn = VSN(inputSize=inputSize,numInputs=1, hiddenSize=hiddenSize,dropout=dropout)
        
        self.lstm = nn.LSTM(inputSize=hiddenSize,hiddenSize=hiddenSize,num_layers=encodlays, batch_first=True,dropout=dropout)
        
        self.grn = GRN(inputSize=hiddenSize,hiddenSize=hiddenSize,outputSize=hiddenSize,dropout=dropout)
        self.attention = MHA(DModel=hiddenSize,numHeads=numHeads,dropout=dropout)
        
        self.grn2 = GRN(inputSize=hiddenSize,hiddenSize=hiddenSize,outputSize=hiddenSize,dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hiddenSize, hiddenSize * 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenSize * 4, hiddenSize)
        )
        self.grn3 = GRN(inputSize=hiddenSize,hiddenSize=hiddenSize,outputSize=hiddenSize,dropout=dropout)
        self.outputProj = nn.Linear(hiddenSize, outputSize)
        self.normaliz = nn.LayerNorm(hiddenSize)
        
    
    def forward(self, x):
        bs, seq_len, z = x.shape
        x = x.view(bs * seq_len, 1, self.inputSize)
        selected,vweight = self.vsn(x)
        selected = selected.view(bs, seq_len, self.hiddenSize)
        lstm_out, z = self.lstm(selected)
        lstm_out = self.grn(lstm_out)
        atoipt, weights = self.attention(lstm_out, lstm_out, lstm_out)
        atoipt = self.grn2(atoipt)
        combined = self.normaliz(lstm_out + atoipt)
        ff_out = self.feed_forward(combined)
        combined = self.normaliz(combined + ff_out)
        final = self.grn3(combined)
        pooled = torch.mean(final, dim=1)
        output = self.outputProj(pooled)
        x = {'variable_selection_weights': vweight,'weights': weights}
        
        return output, x
    
    def predict(self, x) :
        self.eval()
        with torch.no_grad():
            output, z = self.forward(x)
        return output.cpu().numpy()


class TFTForecaster:
    def __init__(self,
                 inputSize = 1,
                 hiddenSize = 64,
                 numHeads = 4,
                 encodlays = 2,
                 outputSize = 5,
                 dropout = 0.1,
                 learning_rate = 0.001,
                 device = None):
        

        self.device = torch.device(device)
        
        self.model = TFT(inputSize=inputSize,hiddenSize=hiddenSize,numHeads=numHeads,encodlays=encodlays,outputSize=outputSize,dropout=dropout).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        
    
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
            
            indices = torch.randpositionalembeddingrm(len(Xtrain))
            
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
            self.trainLosses.appositionalembeddingnd(avgTrainLoss)
            
            if validation:
                self.model.eval()
                with torch.no_grad():
                    valOutput,z = self.model(Xval)
                    
                    val_loss = self.criterion(valOutput, yval).item()
                    self.valLosses.appositionalembeddingnd(val_loss)
                    

        
        return {'train_loss': self.trainLosses,'val_loss': self.valLosses}
    
    def predict(self, X):
        return self.model.predict(torch.FloatTensor(X).unsqueeze(-1).to(self.device))
    