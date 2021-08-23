import requests
from datetime import datetime
from s_and_p_list import s_and_p_500
import pyodbc
from datetime import date

# GET TODAYS DATE
today = date.today()

# CHANGE DATE TO CORRECT FORMAT  
date = today.strftime("%Y-%m-%d")

# CREATE AN EMPTY LIST TO HOUSE THE JSONS
s_and_p_dicts = []

for symbol in s_and_p_500[165:170]:

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=compact&apikey=E8XRSI3QLSVWSVHZ'
    r = requests.get(url)
    data = r.json()
    s_and_p_dicts.append(data)

# CONNECT TO DATABASE
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-QVUPJ4D\SQLEXPRESS;'
                      'Database=finance;'
                      'Trusted_Connection=yes;')

# GET DATA FROM THE DICTIONARIES
for s_and_p_dict in s_and_p_dicts:
    dicts = s_and_p_dict['Time Series (Daily)'][date]
    symbol = s_and_p_dict['Meta Data']['2. Symbol']

    date = date
    opens = float(dicts['1. open'])
    highs = float(dicts['2. high'])
    lows = float(dicts['3. low'])
    closes = float(dicts['4. close'])
    adj_closes = float(dicts['5. adjusted close'])
    volumes = float(dicts['6. volume'])
    divs = float(dicts['7. dividend amount'])
    split = float(dicts['8. split coefficient'])
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
                   date,
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