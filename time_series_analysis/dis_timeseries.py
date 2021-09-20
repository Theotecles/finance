from datetime import datetime
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error

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
    ORDER BY data_date ASC
    '''

# PULL IN THE DATA USING PANDAS
dis_df = pd.read_sql(query, conn)
print(dis_df.head())

# PUT THE DATES INTO THE CORRECT FORMAT FOR MATPLOT
dis_dates = []
for date in dis_df['data_date']:
    dis_dates.append(datetime.strptime(date, '%Y-%m-%d'))

dis_df['data_date'] = dis_dates

# CHECK DATA TYPES
print(dis_df.dtypes)

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
dis_df.index = pd.DatetimeIndex(dis_df.index).to_period('D')
print(dis_df.index)
print(dis_df.head())

# EVALUATE COMBINATIONS OF P, D AND Q VALUES FOR AN ARIMA MODEL
def evaluate_models(dataset, p_values, d_values, q_values):
    for p in p_values:
        for d in d_values:
            for q in q_values:
                model = ARIMA(dataset, order=(p, d, q))
                model_fit = model.fit()
                print(model_fit.summary())

# SPLIT THE DATA INTO TRAINING AND TEST SET
print(len(dis_df))
train = dis_df.head(149)
test = dis_df.tail(50)
print(train.shape)
print(test.shape)

# EVALUATE PARAMETERS
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, p_values, d_values, q_values)

#1, 1, 0 HAS BEST AIC WITH 774.237
# FIT THE MODEL
model = ARIMA(train, order=(1, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# CREATE THE TEST PREDICTIONS
print(test.head(5))
print(test.tail(5))
start_index = datetime(2021, 7, 7)
end_index = datetime(2021, 9, 17)
test['predictions'] = model_fit.predict(start=start_index, end=end_index)
print(test.head(5))
print(test.tail(5))

# PLOT THE DATA
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(dis_dates, dis_df['adjusted_close'], c='red', alpha=0.6)
ax.plot(dis_dates[149:], test['predictions'], c='blue', alpha=0.6)

# FORMAT PLOT
ax.set_title(f"Daily Adjusted Close Data \n(DIS)", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(dis_df['adjusted_close']), max(dis_df['adjusted_close']))

plt.show()
