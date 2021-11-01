from datetime import datetime
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error
from finance import evaluate_models
from finance import evaluate_models_rmse

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
    WHERE symbol = 'BA'
    AND data_date >= '2020-01-01'
    ORDER BY data_date ASC
    '''

# PULL IN THE DATA USING PANDAS
ba_df = pd.read_sql(query, conn)
print(ba_df.head())

# PUT THE DATES INTO THE CORRECT FORMAT FOR MATPLOT
ba_dates = []
for date in ba_df['data_date']:
    ba_dates.append(datetime.strptime(date, '%Y-%m-%d'))

ba_df['data_date'] = ba_dates
symbol = ba_df['symbol'][1]

# PLOT THE DATA
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(ba_dates, ba_df['adjusted_close'], c='red', alpha=0.6)

# FORMAT PLOT
ax.set_title(f"Daily Adjusted Close Data \n({symbol})", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(ba_df['adjusted_close']), max(ba_df['adjusted_close']))

plt.show()

# CONVERT DATAFRAME TO A DATETIME INDEX
ba_df = ba_df.drop(['symbol'], axis=1)
ba_df['data_date'] = pd.to_datetime(ba_df['data_date'])
ba_df.set_index('data_date', inplace=True)
ba_df.index = pd.DatetimeIndex(ba_df.index).to_period('D')

# EVALUATE COMBINATIONS OF P, D AND Q VALUES FOR AN ARIMA MODEL
# SPLIT THE DATA INTO TRAINING AND TEST SET
train = ba_df.head(round(len(ba_df)*.75))
test = ba_df.tail(round(len(ba_df)*.25))
print(test.head())
print(test.tail())

# EVALUATE PARAMETERS
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train, p_values, d_values, q_values)
start_index = datetime(2021, 4, 28)
end_index = datetime(2021, 10, 5)
evaluate_models_rmse(train, p_values, d_values, q_values, actuals=test['adjusted_close'], start_index=start_index, end_index=end_index)

# RMSE for Model (8, 2, 2): 9.595829938407833
# FIT THE MODEL
model = ARIMA(train, order=(8, 2, 2))
model_fit = model.fit()
print(model_fit.summary())

# CREATE THE TEST PREDICTIONS
predictions = model_fit.predict(start=start_index, end=end_index)
test['predictions'] = predictions
date_array = test.index

# PLOT THE DATA
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(ba_dates, ba_df['adjusted_close'], c='red', alpha=0.6)
ax.plot(date_array, test['predictions'], c='blue', alpha=0.6)

# FORMAT PLOT
ax.set_title(f"Daily Adjusted Close Data \n({symbol})", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(ba_df['adjusted_close']), max(ba_df['adjusted_close']))

plt.show()

# FIT AND PLOT FINAL MODEL
model = ARIMA(ba_df, order=(8, 2, 2))
model_fit = model.fit()
print(model_fit.summary())
start_index = datetime(2021, 10, 6)
end_index = datetime(2022, 10, 6)
projections = model_fit.predict(start=start_index, end=end_index)
date_array = projections.index

# PLOT THE DATA
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(ba_dates, ba_df['adjusted_close'], c='red', alpha=0.6)
ax.plot(date_array, projections, c='blue', alpha=0.6)

# FORMAT PLOT
ax.set_title(f"Daily Adjusted Close Data \n({symbol})", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(ba_df['adjusted_close']), max(ba_df['adjusted_close']))

plt.show()