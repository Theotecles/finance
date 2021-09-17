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
# FIT THE MODEL
model = ARIMA(dis_df, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# EVALUATE AN ARIMA MODEL FOR A GIVEN ORDER (P, D, Q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

test_rmse = evaluate_arima_model(dis_df, arima_order=(1, 1, 1))
print(test_rmse)

# EVALUATE COMBINATIONS OF P, D AND Q VALUES FOR AN ARIMA MODEL
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# EVALUATE PARAMETERS
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(dis_df, p_values, d_values, q_values)
