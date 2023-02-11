#
#  Stock price prediction  according to   https://www.youtube.com/watch?v=dKBKNOn3gCE
#

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("***!!! New run !!!***")

tsla_df = yf.download('TSLA', start='2020-01-01', end='2023-02-02', progress=True)


print(tsla_df)

# Get  close coulumn
df = pd.DataFrame(tsla_df.iloc[:, 3:4].values)
df.columns = ['close']

print(df.info())

df["returns"] = df.close.pct_change()
df["log_returns"] = np.log(1+ df["returns"])


#plt.plot(df.log_returns)
#plt.show()

df.dropna(inplace=True)
X = df[["close", "log_returns"]].values

#print(X)

scaler = MinMaxScaler(feature_range = (0, 1)).fit(X)
X_scaled = scaler.transform(X)

print(X_scaled[:5])

y = [x[0] for x in X_scaled]

#print(y[:5])

split = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split]
X_test = X_scaled[split : len(X_scaled)]

y_train = y[:split]
y_test = y[split : len(y)]

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

n = 3

Xtrain = []
ytrain = []
Xtest = []
ytest = []

for i in range(n, len(X_train)):
    Xtrain.append(X_train[i-n : i, :X_train.shape[1]])
    ytrain.append(y_train[i])

for i in range(n, len(X_test)):
    Xtest.append(X_test[i-n : i, :X_test.shape[1]])
    ytest.append(y_train[i])


Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))


Xtest, ytest = (np.array(Xtest), np.array(ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
model.add(tf.keras.layers.Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")
model.fit(Xtrain, ytrain, epochs = 250 , validation_data = (Xtest, ytest), batch_size = 16, verbose = 1)


print(model.summary())


at = []
at.append(X_test[len(X_test) - 4 : len(X_test)-1, :X_test.shape[1]])
at = np.array(at)

print(at)

at = np.reshape(at, (at.shape[0], at.shape[1], at.shape[2]))

a = model.predict(at)

a = np.c_[a, np.zeros(a.shape)]

a = scaler.inverse_transform(a)

print(at)
print(a)
a = [x[0] for x in a]

print("************************************************************")
print("AAAA")
print(at)
print("BBB")
print(a)
exit()

#print(close_prices)
stock_data_len = close_prices.size


sc = MinMaxScaler(feature_range = (0, 1))
close_prices_scaled = sc.fit_transform(close_prices)


features = []
labels = []

for i in range(60, stock_data_len):
    features.append(close_prices_scaled[i-60:i, 0])
    labels.append(close_prices_scaled[i, 0])


features = np.array(features)
labels = np.array(labels)

features = np.reshape(features, (features.shape[0], features.shape[1], 1))
print("AAAA")
print(labels.shape)
print(features.shape)
print(stock_data_len)



model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (features.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units = 50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units = 1)
])

print(model.summary())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

from time import time
start = time()
history = model.fit(features, labels, epochs = 50, batch_size = 32, verbose = 1)
end = time()



print('Total training time {} seconds'.format(end - start))

tsla_test = yf.download('TSLA', start='2022-04-01', end='2023-02-02', progress=True)

test_stock_data_processed = tsla_test.iloc[:, 1:2].values

print(test_stock_data_processed.shape)

all_stock_data = pd.concat((tsla_df['Close'], tsla_df['Close']), axis = 0)

print(all_stock_data)
inputs = all_stock_data[len(all_stock_data) - len(tsla_test) - 60:].values
print("Inputs")
print(inputs)
print(inputs[inputs.size - 61:inputs.size-1])
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
#print("INPUTS SHAPE")
#print("------------")
#print(inputs.shape)

#print(inputs[100:160, 0])

#print("------------")

#print(inputs[100:160, 0])
X_test = []
print("!!!!")
print(inputs.size)
X_test.append(inputs[inputs.size - 61:inputs.size-1, 0])
print(X_test)

X_test = np.array(X_test)
print("before reshape")
print(X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print("INPUTS")
print(X_test.shape)
print("PR")
print(predicted_stock_price.shape)
print(predicted_stock_price)




