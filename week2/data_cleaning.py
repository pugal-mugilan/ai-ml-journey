import pandas as pd
import numpy as np
from scipy import stats

inf = pd.read_csv('data/INFY.csv')
wipro = pd.read_csv('data/WIPRO.csv')
tcs = pd.read_csv('data/TCS.csv')
axis = pd.read_csv('data/AXISBANK.csv')
hdfc = pd.read_csv('data/HDFCBANK.csv')
icici = pd.read_csv('data/ICICIBANK.csv')

df = pd.concat([inf, wipro, tcs, axis, hdfc, icici])
df['Date'] = pd.to_datetime(df['Date'])
df['Symbol'] = df['Symbol'].replace({
    'INFOSYSTCH':'INFY',
    'UTIBANK':'AXISBANK',
})
df = df.drop(columns=['Series','Open','High','Low','Volume','Turnover','Last','VWAP','Trades','Deliverable Volume','%Deliverble']).assign(Sector= lambda df: np.where(df['Symbol'].isin(['INFY' ,'WIPRO', 'TCS']),'IT','BANK'))
df['daily_returns'] = (df['Close'] - df['Prev Close'])/ df['Prev Close'] * 100

mean_value = df.groupby('Sector').agg({'daily_returns':'mean'}).reset_index()
print(mean_value)
print('---------------------------------------')
p_value = stats.ttest_ind(df['daily_returns'][df['Sector'] == 'IT'],df['daily_returns'][df['Sector'] == 'BANK']).pvalue
print(f"p-value = {p_value}")
