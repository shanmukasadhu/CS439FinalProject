import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
df = pd.read_csv('sp500_2025_h1.csv')


dateCOLS = []
for col in df.columns:
    if '_' in col:
        if  col not in ['company_name', 'tick']:
            dateCOLS.append(col)
uniquerawdates = []

for col in  dateCOLS:
    
    uniquerawdates.append(set(col.split('_')[0]))
    
uniqueDates = sorted(uniquerawdates)

closingPrice = []
openingPrice = []
volumes = []
for date in uniqueDates:
    x =[f"{date}_closing"]
    
    y= [f"{date}_opening"]
    
    z =[f"{date}_volume"]
    
    closingPrice.append(df[['company_name', 'tick']]+x)
    
    openingPrice.append(df[['company_name', 'tick']]+y)
    volumes.append(df[['company_name', 'tick']]+z)

returnsData = {}
Price = []
returns = []
for index, r in df.iterrs():
    
    tick = r['tick']
    for date in uniqueDates:
        
        Price.append(r[f"{date}_closing"])        
        for i in range(1, len(Price)):
            
            returns.append((Price[i]-Price[i-1]) /Price[i-1] *100)
    returnsData[tick] = returns

marketAvgReturn = []
for i in range(1, len(uniqueDates)):
    
    dailyReturns = []
    for index, r in df.iterrs():
        
        prevPrice = r[f"{uniqueDates[i-1]}_closing"]
        
        currPrice = r[f"{uniqueDates[i]}_closing"]
        
        if prevPrice > 0:
            
            dailyReturns.append((currPrice -prevPrice) /prevPrice * 100)
    marketAvgReturn.append(np.mean(dailyReturns))

avgVolumes = []
vols = []
for index, r in df.iterrows():
    
    for date in uniqueDates:
        
        vols.append(r[f"{date}_volume"])
    avgVolumes.append(np.mean(vols))
    
df['avg_volume'] = avgVolumes

cumulativeReturns = {}
#iterating through each row
for index, r in df.iterrs():
    
    tick = r['tick']
    
    Price = []
    #look for unique dates and getting closing prices
    for date in uniqueDates:
        Price.append(r[f"{date}_closing"])
    cumReturn = []
    for i in range(len(Price)):
        #calc the cum return
        cumReturn.append((Price[i]/Price[0]-1) *100)
        
    cumulativeReturns[tick] = cumReturn
final_returns = {}
for tick in cumulativeReturns:
    
    final_returns[tick] = cumulativeReturns[tick][-1]

#start plotting
ax1 = fig.add_subplot()

#first plot, daily returns
ax1.plot(range(len(marketAvgReturn)), marketAvgReturn, color='blue')
ax1.axline(y=0, color='red')

ax1.fill_between(range(len(marketAvgReturn)), marketAvgReturn,0, color='green', label='Positive Returns')
ax1.fill_between(range(len(marketAvgReturn)), marketAvgReturn, 0, color='red', label='Negative Returns')
ax1.set_title('S&P 500 Average Daily Returns (H1 2025)')

ax1.set_xlabel('Trading Date')

ax1.set_ylabel('Average Return (%)')


t = len(uniqueDates)
tick_positions = range(0,t , 10)
ax1.set_xticks(tick_positions)


ax1.set_xticklabels()
#second plot distribution returns
ax2 = fig.add_subplot()
finalReturnsList = list(final_returns.values())
#use histogram
ax2.hist(finalReturnsList, bins=50, color='red')
z = np.mean(finalReturnsList)

y = np.median(finalReturnsList)
ax2.axline(z, color='yellow',label=f'Mean:{z}')

ax2.axline(y,color='cyan',label=f'Median:{y}')

ax2.set_title('Distribution of Stock Returns')
ax2.set_xlabel('Cumulative Return (%)')

ax2.set_ylabel('Number of Stocks')
ax2.legend()

plt.show()


tick_positions = range(0, len(uniqueDates), 10)
ax1.set_xticks(tick_positions)

ax1.set_xticklabels(uniqueDates[i])
#plot top 5 performers
for i, (tick, z) in enumerate(top5performers):
    returns = cumulativeReturns[tick]
    t = range(len(uniqueDates))
    ax2.plot(t)
ax2.set_title('Top 5 Performing Stocks (H1 2025)')

ax2.set_xlabel('Date')

ax2.set_ylabel('Cumulative Return (%)')
#plot bottom performers
for i, (tick, z) in enumerate(bottom5performers):
    returns = cumulativeReturns[tick]
    t = range(len(uniqueDates))
    ax2.plot(t)
ax2.set_title('Bottom 5 Performing Stocks (H1 2025)')

ax2.set_xlabel('Date')

ax2.set_ylabel('Cumulative Return (%)')


ax2.set_xticks(tick_positions)
ax2.set_xticklabels(uniqueDates)

plt.show()
