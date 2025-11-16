import pandas as pd

data = pd.read_csv('Data/trade_log.csv')
print(data['Profit/Loss'].sum())