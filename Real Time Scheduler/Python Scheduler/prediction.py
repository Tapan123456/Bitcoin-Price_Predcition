# from keras.models import load_model
# import pandas as pd
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import mean_squared_error
# model = load_model("Bitcoin_model.h5")


# df1 = pd.read_csv('prediction_input11.csv')
# #df1 = df1.drop(['nos'], axis=1)
# values = df1.values

# df1.set_index('Date', inplace=True)
# cols = df1.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# df1 = df1[cols]

# ####################
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     '''
#     Converts the original dataframe to a format which contains
#     lag shifted values of inputs which can be used as input
#     to the LSTM
#     '''
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg

# ##############################

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(df1.values)
# n_hours = 3
# n_features = 5
# n_obs = n_hours * n_features
# reframed = series_to_supervised(scaled, n_hours, 1)
# reframed = reframed.drop(reframed.columns[-4:], axis=1)
# values = reframed.values
# #n=1
# test = values[:,:]
# #print(test.shape)   
# test_X, test_y = test[:,:n_obs], test[:, -n_features]

# test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# yhat = model.predict(test_X)
# # print(yhat.shape)
# # temp=test_X[:, -4:]
# # print(temp.shape)
# test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# inv_yhat = np.concatenate((yhat, test_X[:, -4:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = str(inv_yhat[:, 0])
# print(inv_yhat)
# # df_sol = pd.DataFrame({'forecast': inv_yhat})

# from datetime import date
# import datetime
# today = date.today()
# tomorrow = today + datetime.timedelta(days=1)

# from exchanges.bitfinex import Bitfinex
# price = Bitfinex().get_current_price()
# df=pd.read_csv("input_to_chart.csv")
# df = df.drop(df.columns[0], axis=1)
# #df.append({'Date' : today , 'Actual_Price' : price} , ignore_index=True)
# df.iloc[-1, df.columns.get_loc('Actual_Price')] = price
# inv_yhat = inv_yhat[1:-1]
# new_row = {'Date': tomorrow , 'Predicted_Price' : inv_yhat}
# df=df.append(new_row, ignore_index=True)
# df.reset_index(drop=True, inplace=True)
# #df = df.drop(df.columns[0], axis=1)
# print(df)
# df.to_csv('input_to_chart.csv')

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


loaded_model = load_model("Bitcoin_model.h5")

df1 = pd.read_csv('prediction_input11.csv')
df1 = df1.drop(['nos'], axis=1)
df1 = df1.tail(4)
#print(df1)
values = df1.values
# specify columns to plot
groups = [1, 2, 3, 4, 5]
i = 1
# plot each column
plt.figure()
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(df1.columns[group], y=0.5, loc='right')
#     i += 1

df1.set_index('Date', inplace=True)

cols = df1.columns.tolist()
cols = cols[-1:] + cols[:-1]
df1 = df1[cols]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    Converts the original dataframe to a format which contains
    lag shifted values of inputs which can be used as input
    to the LSTM
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df1.values)
n_hours = 3
n_features = 5
n_obs = n_hours*n_features
reframed = series_to_supervised(scaled, n_hours, 1)
reframed = reframed.drop(reframed.columns[-4:], axis=1)
values = reframed.values



test = values[:, :]

test_X = test[:, :n_obs]


# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

yhat = loaded_model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = str(inv_yhat[:, 0])
print(inv_yhat)

# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, test_X[:, -4:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
#print(inv_y)

#print(inv_yhat.shape)
#df = pd.DataFrame({'forecast': inv_yhat})

from datetime import date
import datetime
today = date.today()
tomorrow = today + datetime.timedelta(days=1)

from exchanges.bitfinex import Bitfinex
price = Bitfinex().get_current_price()

df=pd.read_csv("input_to_chart.csv")
df = df.drop(df.columns[0], axis=1)
df.iloc[-1, df.columns.get_loc('Actual_Price')] = price
inv_yhat = inv_yhat[1:-1]
new_row = {'Date': tomorrow , 'Predicted_Price' : inv_yhat}
df=df.append(new_row, ignore_index=True)
#print(df)
# df.reset_index(drop=True, inplace=True)
# print(df)
df.to_csv('input_to_chart.csv')
