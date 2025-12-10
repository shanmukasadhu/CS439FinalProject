import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, Dmodel, leng= 5000):
        super(PositionalEncoding, self).__init__()
        
        positionalembedding = torch.zeros(leng, Dmodel)
        position = torch.arange(0, leng, dtypositionalembedding=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, Dmodel, 2).float() * (-math.log(10000.0) /Dmodel))
        
        positionalembedding[:, 0::2] = torch.sin(position * divTerm)
        positionalembedding[:, 1::2] = torch.cos(position * divTerm)
        
        positionalembedding = positionalembedding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.positionalembedding[:,:x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, Dmodel, heads, dropout):
        super(MultiHeadAttention, self).__init__()
        
        
        self.Dmodel = Dmodel
        self.heads = heads
        self.kconstant = Dmodel // heads
        
        self.query = nn.Linear(Dmodel, Dmodel)
        self.key = nn.Linear(Dmodel, Dmodel)
        self.value = nn.Linear(Dmodel, Dmodel)
        
        self.output = nn.Linear(Dmodel, Dmodel)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask):
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.kconstant)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask):
        bs = x.size(0)
        
        Q = self.query(x) 
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(bs, -1,self.heads, self.kconstant).transpose(1, 2)
        K = K.view(bs, -1, self.heads,self.kconstant).transpose(1, 2)
        
        V = V.view(bs, -1,self.heads, self.kconstant).transpose(1, 2)
        attnOutput,z = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attnOutput = attnOutput.transpose(1, 2).contiguous()
        attnOutput = attnOutput.view(bs, -1, self.Dmodel)
        
        output = self.output(attnOutput)
        
        return output


class FFC(nn.Module):
    def __init__(self, Dmodel, dfeedforward, dropout = 0.1):
        super(FFC, self).__init__()
        
        self.linear1 = nn.Linear(Dmodel, dfeedforward)
        
        self.linear2 = nn.Linear(dfeedforward, Dmodel)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, Dmodel, heads, dfeedforward, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.selfAttention = MultiHeadAttention(Dmodel, heads, dropout)
        self.feed_forward = FFC(Dmodel, dfeedforward, dropout)
        
        self.norm1 = nn.LayerNorm(Dmodel)
        self.norm2 = nn.LayerNorm(Dmodel)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attnOutput = self.selfAttention(x, mask)
        
        x = self.norm1(x +self.dropout1(attnOutput))
        
        ffOutput = self.feed_forward(x)
        x = self.norm2(x +self.dropout2(ffOutput))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,inputSize= 1,Dmodel= 64,heads= 4,numLayers= 2,dfeedforward= 256,output_size= 5,dropout = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.inputSize = inputSize
        self.Dmodel = Dmodel
        self.heads = heads
        self.numLayers = numLayers
        self.inputProj = nn.Linear(inputSize, Dmodel)
        self.posCncoder = PositionalEncoding(Dmodel)
        
        self.encoderLays = nn.ModuleList([TransformerEncoderLayer(Dmodel, heads, dfeedforward, dropout)for z in range(numLayers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(Dmodel, output_size)
        
    
    def forward(self, x, mask):
        x = self.inputProj(x) 
        
        x = self.posCncoder(x)
        x = self.dropout(x)
        
        for layer in self.encoderLays:
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
    def __init__(self,inputSize= 1,Dmodel= 64,heads= 4,numLayers= 2,dfeedforward= 256,output_size= 5,dropout = 0.1,learning_rate = 0.001,device = None):
        

        self.device = torch.device(device)
        
        self.model = TransformerEncoder(inputSize=inputSize,Dmodel=Dmodel,heads=heads,numLayers=numLayers,dfeedforward=dfeedforward,output_size=output_size,dropout=dropout).to(self.device)
        
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
    