#
#  Stock price prediction multivariate lstm  according to  https://www.youtube.com/watch?v=tepxdcepTbY
#

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


df_symbol = yf.download('MSFT', start='2019-01-01', end='2023-02-02', progress=True)
#print(df_symbol)

cols = list(df_symbol)[1:6]

df_for_training = df_symbol[cols].astype(float)

print(df_for_training)
a
#df_for_plot=df_for_training.tail(5000)
#df_for_plot.plot.line()
#plt.show()

scaler = MinMaxScaler(feature_range = (0, 1)).fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future = 1

n_past = 14

for i in range (n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future -1 :i + n_future, 0:])
    

print("*** End ****")
