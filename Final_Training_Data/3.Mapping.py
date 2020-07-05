import pandas as pd
import datetime
from datetime import date
data = pd.read_csv('Sentiment_Data_Output_of_Step2.csv')
prices = pd.read_csv('BTCPrices.csv')

prices_closing = []
prices_opening = []
prices_high = []
prices_low = []

for ind in data.index:
    date = data['Date'][ind]
    #print(date)
    #sometimes in excel it may show  date as 13/01/2020 but in dataframe it is 2020-01-13
    date1 = date.split('-')
    
    # dataframe loads date as string , so convert string to date , as it is easier to groupby 
    year = int(date1[0])
    month = int(date1[1])
    day = int(date1[2])

    #calculating next day
    tomorrow = datetime.date(year, month, day) + datetime.timedelta(days=1)

    #matching previous day tweet with next day price , prices date format after loading yyyy-mm-dd
    cp = prices.loc[prices['Date'] == str(tomorrow),'Closing Price (USD)'].values
    op = prices.loc[prices['Date'] == str(tomorrow),'24h Open (USD)'].values
    hp = prices.loc[prices['Date'] == str(tomorrow),'24h High (USD)'].values
    lp = prices.loc[prices['Date'] == str(tomorrow),'24h Low (USD)'].values

    prices_closing.append(cp)
    prices_opening.append(op)
    prices_high.append(hp)
    prices_low.append(lp)

data['Closing_price'] = prices_closing
data['Opening_price'] = prices_opening
data['High_price'] = prices_high
data['Low_price'] = prices_low

data.to_csv('Final_Output_After_Mapping.csv')
