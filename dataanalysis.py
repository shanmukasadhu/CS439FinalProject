import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
df = pd.read_csv('sp500_2025_h1.csv')

date_cols = [col for col in df.columns if '_' in col and col not in ['company_name', 'ticker']]
unique_dates_raw = list(set([col.split('_')[0] for col in date_cols]))

from datetime import datetime
unique_dates = sorted(unique_dates_raw, key=lambda x: datetime.strptime(x, '%d-%m-%Y'))


closing_prices = df[['company_name', 'ticker'] + [f"{date}_closing" for date in unique_dates]]
opening_prices = df[['company_name', 'ticker'] + [f"{date}_opening" for date in unique_dates]]
volumes = df[['company_name', 'ticker'] + [f"{date}_volume" for date in unique_dates]]

returns_data = {}
for idx, row in df.iterrows():
    ticker = row['ticker']
    prices = [row[f"{date}_closing"] for date in unique_dates]
    returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
    returns_data[ticker] = returns

market_avg_return = []
for i in range(1, len(unique_dates)):
    daily_returns = []
    for idx, row in df.iterrows():
        prev_price = row[f"{unique_dates[i-1]}_closing"]
        curr_price = row[f"{unique_dates[i]}_closing"]
        if prev_price > 0:
            daily_returns.append((curr_price - prev_price) / prev_price * 100)
    market_avg_return.append(np.mean(daily_returns))

avg_volumes = []
for idx, row in df.iterrows():
    vols = [row[f"{date}_volume"] for date in unique_dates]
    avg_volumes.append(np.mean(vols))
df['avg_volume'] = avg_volumes
top_10_volume = df.nlargest(10, 'avg_volume')

cumulative_returns = {}
for idx, row in df.iterrows():
    ticker = row['ticker']
    prices = [row[f"{date}_closing"] for date in unique_dates]
    cum_return = [(prices[i] / prices[0] - 1) * 100 for i in range(len(prices))]
    cumulative_returns[ticker] = cum_return
final_returns = {ticker: cumulative_returns[ticker][-1] for ticker in cumulative_returns}
top_5_performers = sorted(final_returns.items(), key=lambda x: x[1], reverse=True)[:5]
bottom_5_performers = sorted(final_returns.items(), key=lambda x: x[1])[:5]
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(range(len(market_avg_return)), market_avg_return, linewidth=2.5, color='#2E86AB', marker='o', markersize=4)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax1.fill_between(range(len(market_avg_return)), market_avg_return, 0, 
                  where=np.array(market_avg_return) > 0, alpha=0.3, color='green', label='Positive Returns')
ax1.fill_between(range(len(market_avg_return)), market_avg_return, 0, 
                  where=np.array(market_avg_return) <= 0, alpha=0.3, color='red', label='Negative Returns')
ax1.set_title('S&P 500 Average Daily Returns (H1 2025)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Trading Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Return (%)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)



tick_positions = range(0, len(unique_dates[1:]), 10)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels([unique_dates[1:][i] for i in tick_positions], rotation=45, ha='right')

ax2 = fig.add_subplot(gs[1, 0])
final_returns_list = list(final_returns.values())
ax2.hist(final_returns_list, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(final_returns_list), color='yellow', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_returns_list):.2f}%')
ax2.axvline(np.median(final_returns_list), color='cyan', linestyle='--', linewidth=2, label=f'Median: {np.median(final_returns_list):.2f}%')
ax2.set_title('Distribution of Stock Returns', fontsize=14, fontweight='bold')
ax2.set_xlabel('Cumulative Return (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Stocks', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot1_market_overview.png', dpi=300, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
fig.subplots_adjust(hspace=0.4)

colors_top = plt.cm.Greens(np.linspace(0.4, 0.9, 5))
for i, (ticker, _) in enumerate(top_5_performers):
    cum_ret = cumulative_returns[ticker]
    ax1.plot(range(len(unique_dates)), cum_ret, marker='o', linewidth=2.5, label=ticker, color=colors_top[i], markersize=5)
ax1.set_title('Top 5 Performing Stocks (H1 2025)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.4)
tick_positions = range(0, len(unique_dates), 10)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels([unique_dates[i] for i in tick_positions], rotation=45, ha='right')

colors_bottom = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
for i, (ticker, _) in enumerate(bottom_5_performers):
    cum_ret = cumulative_returns[ticker]
    ax2.plot(range(len(unique_dates)), cum_ret, marker='o', linewidth=2.5, label=ticker, color=colors_bottom[i], markersize=5)
ax2.set_title('Bottom 5 Performing Stocks (H1 2025)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.4)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([unique_dates[i] for i in tick_positions], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plot2_top_bottom_performers.png', dpi=300, bbox_inches='tight')
plt.show()



all_volatilities = []
all_returns = []
all_tickers = []
for ticker in returns_data:
    vol = np.std(returns_data[ticker])
    ret = final_returns[ticker]
    all_volatilities.append(vol)
    all_returns.append(ret)
    all_tickers.append(ticker)

scatter = ax1.scatter(all_volatilities, all_returns, c=all_returns, 
                     cmap='RdYlGn', s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
ax1.set_title('Risk-Return Profile (All Stocks)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Volatility (Std Dev of Daily Returns %)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cumulative Return (%)', fontsize=11, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.axvline(x=np.mean(all_volatilities), color='blue', linestyle='--', alpha=0.5, label='Mean Volatility')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
plt.colorbar(scatter, ax=ax1, label='Return (%)')
plt.savefig('plot4_risk_return.png', dpi=300, bbox_inches='tight')
plt.show()
