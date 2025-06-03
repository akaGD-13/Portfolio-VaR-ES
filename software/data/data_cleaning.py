import pandas as pd

aapl = pd.read_csv('software/data/AAPL-bloomberg.csv', index_col=0, parse_dates=True)
amzn = pd.read_csv('software/data/AMZN-bloomberg.csv', index_col=0, parse_dates=True)

aapl = aapl[aapl.index >= amzn.index[0]]
# print(aapl)
# print(amzn)

combined = pd.DataFrame({
    'AAPL': aapl['PX_LAST'],
    'AMZN': amzn['PX_LAST']
})

print(combined)

combined.to_csv('software/data/portfolio.csv')