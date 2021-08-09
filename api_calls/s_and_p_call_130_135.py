import requests
from datetime import datetime
from s_and_p_list import s_and_p_500
import pyodbc

# CREATE AN EMPTY LIST TO HOUSE THE JSONS
s_and_p_dicts = []

for symbol in s_and_p_500[130:135]:
    print(symbol)

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey=E8XRSI3QLSVWSVHZ'
    r = requests.get(url)
    data = r.json()
    s_and_p_dicts.append(data)

for data in s_and_p_dicts:
    print(data['Meta Data']['2. Symbol'])

# CONNECT TO DATABASE
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-QVUPJ4D\SQLEXPRESS;'
                      'Database=finance;'
                      'Trusted_Connection=yes;')

# GET DATA FROM THE DICTIONARIES
for s_and_p_dict in s_and_p_dicts:
    dicts = s_and_p_dict['Time Series (Daily)']
    symbol = s_and_p_dict['Meta Data']['2. Symbol']

    for date in dicts:
        dates = date
        opens = float(dicts[date]['1. open'])
        highs = float(dicts[date]['2. high'])
        lows = float(dicts[date]['3. low'])
        closes = float(dicts[date]['4. close'])
        adj_closes = float(dicts[date]['5. adjusted close'])
        volumes = float(dicts[date]['6. volume'])
        divs = float(dicts[date]['7. dividend amount'])
        split = float(dicts[date]['8. split coefficient'])
        stock_value_id = f"{symbol}_{date}"

        # SET UP THE SQL CONNECTION AND RUN THE SCRIPT
        cursor = conn.cursor()
        cursor.execute('''
                       INSERT INTO finance.dbo.s_and_p_daily (stock_value_id,
                                                              symbol,
                                                              data_date,
                                                              open_price,
                                                              high_price,
                                                              low_price,
                                                              close_price,
                                                              adjusted_close,
                                                              volume,
                                                              dividend,
                                                              split
                   )
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ''',
                   stock_value_id,
                   symbol,
                   dates,
                   opens,
                   highs,
                   lows,
                   closes,
                   adj_closes,
                   volumes,
                   divs,
                   split
                   )
        conn.commit()