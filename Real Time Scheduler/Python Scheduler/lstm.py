from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


# Import Data
df1 = pd.read_csv('Final_Output.csv')
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
#n_train_hours = 1250
train = values[:, :]

#test = values[n_train_hours:, :]
train_X, train_y = train[:, :n_obs], train[:, -n_features]
#est_X, test_y = test[:, :n_obs], test[:, -n_features]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
#test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

model = Sequential()
model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=28, batch_size=4,
                    verbose=2, shuffle=False)

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
#

# make a prediction
df1 = pd.read_csv('prediction_input.csv')
values = df1.values


df1.set_index('Date', inplace=True)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df1.values)
n_hours = 1
n_features = 4
n_obs = n_hours * n_features
reframed = series_to_supervised(scaled, n_hours, 1)
reframed = reframed.drop(reframed.columns[-4:], axis=1)

values = reframed.values
# print(values)
test_X = values[:, :]
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

yhat = model.predict(test_X)
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

plt.plot(inv_yhat,  label='predcited')
plt.show()
inv_yhat.shape
df_sol = pd.DataFrame({'forecast': inv_yhat})
df_sol.to_csv('Predictions_nosent5.csv')
