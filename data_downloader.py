import pandas as pd
import yfinance as yf
#from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("***!!! New run !!!***")

tsla_df = yf.download('TSLA', start='2020-01-01', end='2023-02-02', progress=True)
#print(tsla_df.info())

print(tsla_df.head())


# Get  close coulumn
close_prices = tsla_df.iloc[:, 3:4].values
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




