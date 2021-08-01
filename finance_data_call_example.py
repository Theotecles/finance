import requests
from datetime import datetime
import matplotlib.pyplot as plt

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=DIS&apikey=MYAPIKEY'
r = requests.get(url)
dis_data = r.json()

# CREATE A DICTIONARY FOR THE FINANCE DATA
symbol = dis_data['Meta Data']['2. Symbol']

# GET THE DATES AND THE HIGHS AND THE LOWS
dis_dicts = dis_data['Time Series (Daily)']
dis_dates, highs, lows = [], [], []
for date in dis_dicts:
    dis_dates.append(datetime.strptime(date, '%Y-%m-%d'))
    highs.append(float(dis_dicts[date]['2. high']))
    lows.append(float(dis_dicts[date]['3. low']))

# DEFINE REVERESE
def reverse(lst):
    """reverese the elements in a list"""
    return [ele for ele in reversed(lst)]

# PLOT THE HIGHS AND LOWS
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(dis_dates, reverse(highs), c='red', alpha=0.6)
ax.plot(dis_dates, reverse(lows), c='blue', alpha=0.6)
ax.fill_between(dis_dates, reverse(highs), reverse(lows), facecolor='blue', alpha=0.15)

# FORMAT PLOT
ax.set_title(f"Daily high and low stock prices - Last 100 Days \nWalt Disney Company ({symbol})", fontsize=24)
ax.set_xlabel('', fontsize=16)
fig.autofmt_xdate()
ax.set_ylabel("Price USD", fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.ylim(min(lows), max(highs))

plt.show()
