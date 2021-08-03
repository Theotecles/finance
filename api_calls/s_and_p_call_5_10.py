import requests
from datetime import datetime
import matplotlib.pyplot as plt
from s_and_p_list import s_and_p_500

s_and_p_dicts = []

for symbol in s_and_p_500[5:10]:
    print(symbol)

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey=MYAPIKEY'
    r = requests.get(url)
    data = r.json()
    s_and_p_dicts.append(data)

for data in s_and_p_dicts:
    print(data['Meta Data']['2. Symbol'])
