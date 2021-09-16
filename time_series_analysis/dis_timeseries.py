from datetime import datetime
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# CONNECT TO DATABASE
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-QVUPJ4D\SQLEXPRESS;'
                      'Database=finance;'
                      'Trusted_Connection=yes;')
# CREATE A CURSOR
cursor = conn.cursor()

# WRITE THE SQL QUERY
query = '''
    SELECT symbol,
       data_date,
       adjusted_close
    FROM finance.dbo.s_and_p_daily
    WHERE symbol = 'DIS'
    AND data_date >= '2020-12-01'
    ORDER BY data_date DESC
    '''

# PULL IN THE DATA USING PANDAS
dis_df = pd.read_sql(query, conn)
print(dis_df.head())

# PUT THE DATES INTO THE CORRECT FORMAT FOR MATPLOT
dis_dates = []
for date in dis_df['data_date']:
    dis_dates.append(datetime.strptime(date, '%Y-%m-%d'))

dis_df['data_date'] = dis_dates

# PLOT THE DATA
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(dis_dates, dis_df['adjusted_close'], c='red', alpha=0.6)

# FORMAT PLOT
ax.set_title(f"Daily Adjusted Close Data \n({dis_df['symbol'][1]})", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(dis_df['adjusted_close']), max(dis_df['adjusted_close']))

plt.show()

# CONVERT DATAFRAME TO A DATETIME INDEX
dis_df = dis_df.drop(['symbol'], axis=1)
dis_df['data_date'] = pd.to_datetime(dis_df['data_date'])
dis_df.set_index('data_date', inplace=True)
print(dis_df.head())
print(np.asarray(dis_df))
# FIT THE MODEL
model = ARIMA(dis_df, order=(1, 1, 1))
model_fit = model.fit()

# MAKE PREDICTIONS
pred = model_fit.predict(len(dis_df), len(dis_df), typ='levels')
