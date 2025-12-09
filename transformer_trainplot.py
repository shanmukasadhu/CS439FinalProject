
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from transformer_model import TransformerForecaster

plt.style.use('seaborn-v0_8-darkgrid')


df = pd.read_csv('sp500_2024_2025_combined.csv')
date_cols = []
for col in df.columns:
    if '_' in col and col not in ['company_name','tik']:
        date_cols.append(col)
        
undateraw = []
for col in date_cols:
    undateraw.append(col.split('_')[0])

from datetime import datetime

def parse_date(date_str):
    return datetime.strptime(date_str, '%d-%m-%Y')

undate = sorted(undateraw, key=parse_date)


lookbackwindow = 20
forecasthorz = 5
tssplit = 0.8

dmod = 64
heads = 2
layers = 1
D_FF = 128
dr = 0.1
lr = 0.001
ep = 100

def prepStock(tik, dats):
    prices = []
    for date in dats:
        priceperdate = tik[f"{date}_closing"].values[0]
        prices.append(priceperdate)
    return np.array(prices)

def create_sequences(data, look, hor):
    samples = []
    eamples = []
    for i in range(len(data)-look-hor+1):
        samples.append(data[i:i+look])
        eamples.append(data[i+look:i+look+hor])
    return np.array(samples), np.array(eamples)

avgvols = []
for index, row in df.iterrows():
    vols = []
    for date in undate:
        vols.append(row[f"{date}_volume"])
    avgvols.append(np.mean(vols))
df['avg_volume'] = avgvols
t5stocks = df.nlargest(5, 'avg_volume')
for index, row in t5stocks.iterrows():
    print(f"  • {row['tik']} ({row['company_name']})")

def evalFor(act, pred, name):
    #mse
    mse = mean_squared_error(act, pred)
    #rmse
    rmse = np.sqrt(mse)
    # mae
    mae = mean_absolute_error(act, pred)
    #mape
    mape = mean_absolute_percentage_error(act, pred) * 100
    print(f"{name}Performance Metrics:")
    print(f"RMSE, {rmse}")
    
    print(f"MAE, {mae}")
    
    print(f"MAPE,{mape}")
    return {'rmse':rmse,'mae': mae,'mape': mape}

dictforrest = {}

for index, stocksrow in t5stocks.iterrows():
    tik = stocksrow['tik']
    company = stocksrow['company_name']
    prices = prepStock(stocksrow.to_frame().T, undate)
    stndScalar = StandardScaler()
    z = prices.reshape(-1, 1)
    normedprices = stndScalar.fit_transform(z).flatten()
    splindex = int(len(normedprices) * tssplit)
    train = normedprices[:splindex]
    test = normedprices[splindex:]

    xtran, ytran = create_sequences(train, lookbackwindow, forecasthorz)
    xtest, ytest = create_sequences(test, lookbackwindow, forecasthorz)
    

    transformer_model = TransformerForecaster(
            input_size=1,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            d_ff=D_FF,
            output_size=FORECAST_HORIZON,
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE
        )
    # trainmodel
    history = model.fit(xtran, ytran,X_val=xtest, y_val=ytest,epochs=epochs,batch_size=bs,verbose=True)
    
    preds = model.predict(xtest)
    invpred = stndScalar.inverse_transform(preds)
    acts = stndScalar.inverse_transform(ytest)
    
    tpo=prices[:splindex]
    tepo=prices[splindex:]
    

    predictions_flat = invpred.flatten()
    acts_flat = acts.flatten()
    metrics = evalFor(acts_flat, predictions_flat, "transformer")
    
    dictforrest[tik] = {'company': company,'prices': prices,'train_prices': tpo,'test_prices': tepo,'predictions': invpred,'acts': acts,'metrics': metrics,'history': history,'split_index': splindex,'model': model}


fig, axes = plt.subplots(len(t5stocks), 1, figsize=(16, 4*len(t5stocks)))
y = dictforrest.items()
for index, (tik, rest) in enumerate():
    ax = axes[index]
    
    ax.plot(range(len(rest['prices'])), rest['prices'], label='act Price')
        
    test_start = rest['split_index'] + lookbackwindow
    for i, pred in enumerate(rest['predictions']):
        
        start_index = test_start + i
        
        pred_range = range(start_index, start_index + forecasthorz)
        
        ax.plot(pred_range, pred, color='red')
    
    ax.set_title(f'{tik}-transformer Forecast RMSE: ${rest["metrics"]["rmse"]} | MAPE: {rest["metrics"]["mape"]}%', fontweight='bold')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Prick')
    ax.legend(loc='upper left')

plt.show()

fig, axes = plt.subplots(len(t5stocks),1,figsize=(16, 20))
l = dictforrest.items()
for index, (tik, rest) in enumerate(l):
    ax = axes[index]
    
    history = rest['history']
    ep_ra=[]
    for i in range(1, len(history['train_loss']) + 1):
        ep_ra.append(i)
    
    ax.plot(ep_ra, history['train_loss'], label='Training Loss', color='blue')
    
    ax.plot(ep_ra, history['val_loss'], label='Validation Loss', color='orange')
    
    ax.set_title(f'{tik} - Training History')
    ax.set_xlabel('Epoch')
    
    ax.set_ylabel('Loss (MSE)')
    
    ax.legend(loc='upper right')

plt.show()

fig, axes = plt.subplots(len(t5stocks),1,figsize=(16, 20))

l = dictforrest.items()
for index, (tik, rest) in enumerate(l):
    ax = axes[index]
    
    test_range = range(len(rest['test_prices']))
    ax.plot(test_range, rest['test_prices'])
    
    for i, pred in enumerate(rest['predictions']):
        start_index = lookbackwindow + i
        #predrange
        pred_range = range(start_index, start_index + forecasthorz)
        #sad
        ax.plot(pred_range, pred, color='red', label='Forecast')
    
    ax.set_title(f'{tik} - Test Set Performance')
    ax.set_xlabel('Test Set Day')
    
    ax.set_ylabel('Price ($)')
    
    ax.legend(loc='upper left')
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

tiks = dictforrest.keys()
tiks=list(tiks)
rmse_values = []
mae_values = []
mape_values = []
for t in tiks:
    rmse_values.append(dictforrest[t]['metrics']['rmse'] )
    
    mae_values.append(dictforrest[t]['metrics']['mae'] )
    
    mape_values.append(dictforrest[t]['metrics']['mape'] )

#Plotting COde for RMSE
ax1.bar(tiks, rmse_values, color="blue")
ax1.set_title('Root Mean Squared Error (RMSE)')

ax1.set_ylabel('RMSE ($)')

#same for mae
ax2.bar(tiks, mae_values, color="red")
ax2.set_title('Mean Absolute Error (MAE)')

ax2.set_ylabel('MAE ($)')


#same for mape
ax3.bar(tiks, mape_values, color="green")
ax3.set_title('Mean Absolute Percentage Error (MAPE)')
ax3.set_ylabel('MAPE (%)')

plt.show()


