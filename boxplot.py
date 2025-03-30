import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define two sets of stock tickers
set1 = ['0001.HK', '0002.HK', '0003.HK', '0005.HK','0006.HK','0011.HK','0016.HK','0027.HK','0066.HK','0101.HK','0175.HK','0267.HK','0316.HK','0386.HK','0388.HK','0669.HK','0688.HK','0700.HK','0762.HK','0823.HK','0857.HK','0883.HK','0939.HK','0941.HK','1038.HK','1044.HK','1093.HK','1177.HK','1398.HK','1928.HK','2020.HK','2313.HK','2318.HK','2319.HK','2382.HK','2388.HK','2628.HK','3988.HK']
set2 = ['600016.ss','600196.ss','600104.ss','600019.ss','600690.ss','600519.ss','600028.ss','600010.ss','600000.ss','600036.ss','600585.ss','600050.ss','600030.ss','600900.ss','600031.ss','600089.ss','600547.ss','601988.ss','601398.ss','600048.ss','601628.ss','601166.ss','601318.ss','601328.ss','601088.ss','601857.ss','601939.ss','601601.ss','601390.ss','601169.ss','600111.ss','601899.ss','601186.ss','601668.ss','601989.ss','601607.ss','600999.ss','600887.ss','600406.ss','600893.ss','600276.ss','600570.ss','601618.ss','601788.ss','601111.ss','600009.ss','600809.ss','600660.ss','600436.ss','600309.ss','601888.ss','600438.ss','600588.ss','600760.ss','600600.ss','600176.ss','600011.ss' ,'600346.ss','600150.ss','600584.ss','600132.ss','600426.ss','601919.ss', '601600.ss','600460.ss','600085.ss','600096.ss','600188.ss','600795.ss','600875.ss','601872.ss','600026.ss','600027.ss','600160.ss','600332.ss','601766.ss','601998.ss','600895.ss','600489.ss','600418.ss','600415.ss','600118.ss','601168.ss','601006.ss','600886.ss','600482.ss','600372.ss','600362.ss','601898.ss','601699.ss','601009.ss','600845.ss','600803.ss','600674.ss','600039.ss','600018.ss','600015.ss']


# Fetch historical data for both sets
data1 = yf.download(set1, start='2020-01-01', end='2023-01-01')
data2 = yf.download(set2, start='2020-01-01', end='2023-01-01')

# Check available columns for data1
print("Data1 columns:", data1.columns)

# Check available columns for data2
print("Data2 columns:", data2.columns)

# Use 'Adj Close' or 'Close', depending on availability
price_data1 = data1['Adj Close'] if 'Adj Close' in data1.columns else data1['Close']
price_data2 = data2['Adj Close'] if 'Adj Close' in data2.columns else data2['Close']

# Calculate daily returns for both sets
returns1 = price_data1.pct_change().dropna()
returns2 = price_data2.pct_change().dropna()

# Stack returns into a single DataFrame with an identifier for each set
combined_returns1 = returns1.stack().reset_index()
combined_returns1.columns = ['Date', 'Stock', 'Return']
combined_returns1['Group'] = 'HSI'

combined_returns2 = returns2.stack().reset_index()
combined_returns2.columns = ['Date', 'Stock', 'Return']
combined_returns2['Group'] = 'SSE180'

# Combine both sets into one DataFrame
combined_returns = pd.concat([combined_returns1, combined_returns2])

# Create a boxplot for the combined returns
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Return', data=combined_returns)
plt.title('Boxplot of Daily Returns for Two Indices Combinations')
plt.xlabel('Index Combinations')
plt.ylabel('Daily Returns')
plt.grid()
plt.show()