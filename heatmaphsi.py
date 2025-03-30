import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the stock symbols and date range
symbols = ['0168.HK','0175.HK','0268.HK','0270.HK','0288.HK','0316.HK','0357.HK','0358.HK','0383.HK','0390.HK','0489.HK','0525.HK','0552.HK','0568.HK','0576.HK','0586.HK','0598.HK','0670.HK','0694.HK','0728.HK','0737.HK','0753.HK','0763.HK','0777.HK','0813.HK','0902.HK','0914.HK','0936.HK','0998.HK','1288.HK','0001.HK','0002.HK','0003.HK','0004.HK','0005.HK','0006.HK','0011.HK','0012.HK','0016.HK','0017.HK','0019.HK','0023.HK','0066.HK','0083.HK','0101.HK','0135.HK','0144.HK','0151.HK','0267.HK','0291.HK','0293.HK','0330.HK','0386.HK','0388.HK','0688.HK','0700.HK','0762.HK','0823.HK','0857.HK','0883.HK','0939.HK','0941.HK','0992.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK','1177.HK','1299.HK','1398.HK','1928.HK','2007.HK','2318.HK','2388.HK','2628.HK','3328.HK','3988.HK','6823.HK','9618.HK','9988.HK']

start_date = '2010-01-01'
end_date = '2020-12-31'#refer to the period selected by the paper

# Fetch the stock data
data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

# Summary statistics for prices
print("Summary Statistics for Prices:\n")
print(data.describe())

# Calculate the correlation matrix
correlation_matrix = data.pct_change().corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
sns.heatmap(correlation_matrix, annot=False, xticklabels=False, yticklabels=False, cmap='crest', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Stocks in HSI')
plt.show()
