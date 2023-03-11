import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dropout, Dense
from pandas_datareader import data as pdr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, pacf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# import yfinance as yfin
# yfin.pdr_override()

# initializing Parameters
start = "1997-01-01"
end = "2021-01-01"
stock_symbols = ["MSFT", "AMD", "AMZN", "GE", "AAPL", "PFE", "KO", "WMT", "COST", "T", "CMCSA", "PEP",
                 "KO", "UL", "BAC", "JPM", "GS", "MS", "NOC", "LMT", "BA", "F", "LUV", "SLB", "NEE", "MCD",
                 "SBAC", "AMT", "JNJ"] + ["^GSPC", "^DJI", "^IXIC", "^NYA", "^RUT", "^N100"] + ["JPY=X"]

commodity_symbols = ["^IRX", "^FVX", "^TNX"] + ["FDCAX", "GMCFX", "FRESX", "FRRSX", "FSLBX", "VDIGX", "SGGDX", "OPGSX"] \
                    + ["CL", "HE", "B", "NBP", "GL"]
# ["GC=F", "SI=F", "HG=F", "CL=F", "NG=F"]
interval = 'd'
n_periods = 176  # 176 days prediction into future
n_future = 1  # number of future stock prices to predict while training
n_lags = 14  # number of past stock price to consider while training
forecasted = []
# causality_data = []
mses = []

# master_data = pdr.get_data_yahoo(stock_symbols + commodity_symbols, start, end, interval=interval)['Adj Close']
# master_data.dropna(inplace=True)
#
# for i in range(len(stock_symbols)):
#     data = master_data.loc[:, [stock_symbols[i]] + commodity_symbols]
    # print(data.shape)
    # symbols = [stock_symbol] + commodity_symbols

    # Getting the data
    # data = pdr.get_data_yahoo(symbols, start, end, interval=interval)['Adj Close']
    # print(data.shape)

    # fill NaN values with back fill
    # data.fillna(method='bfill', inplace=True)
    # data.dropna(inplace=True)

    # normalizing the data
    # scaler = StandardScaler()
    # scaler = scaler.fit(data)
    # data_scaled = scaler.transform(data)
    #
    # trainX = []
    # trainY = []
    #
    # for index in range(n_lags, len(data_scaled) - n_future + 1):
    #     trainX.append(data_scaled[index - n_lags: index, :data_scaled.shape[1]])
    #     trainY.append(data_scaled[index + n_future - 1: index + n_future, 0])
    #
    # trainX, trainY = np.array(trainX), np.array(trainY)
    # print(trainX.shape, trainY.shape)

    # model = Sequential()
    # model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    # model.add(LSTM(32, activation='relu', return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(trainY.shape[1]))
    # model.compile(optimizer='adam', loss='mse')
    # # model.summary()
    #
    # checkpoint = ModelCheckpoint(filepath='model_{0}'.format(stock_symbol), monitor='val_loss',
    #                              verbose = 2, save_best_only = True, mode ='min')
    #
    # history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=2,
    #                     callbacks=[checkpoint])
    #
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()

    # best_model = tf.keras.models.load_model('model_{0}'.format(stock_symbols[i]))
    # forecasted_prices_scaled = best_model.predict(trainX[-n_periods:])
    #
    # forecast_copies = np.repeat(forecasted_prices_scaled, data_scaled.shape[1], axis=-1)
    # forecasted_prices = scaler.inverse_transform(forecast_copies)[:, 0]
    # print(forecasted_prices.shape)

    # forecasted.append(list(forecasted_prices))
    # print(data[-n_periods:].index)
    # print(forecasted_prices.shape, data.iloc[-n_periods:, 0].shape)

    # print("MSE for {0}: ".format(stock_symbols[i]), mean_squared_error(data.iloc[-n_periods:, 0], forecasted_prices))
    # mses.append(mean_squared_error(data.iloc[-n_periods:, 0], forecasted_prices))

    # plt.title(stock_symbol)
    # plt.xlabel('Datetime')
    # plt.ylabel('Adjusted Closing Prices')
    # plt.plot(data[-n_periods:].index, data.iloc[-n_periods:, 0])
    # plt.plot(data[-n_periods:].index, forecasted_prices)
    # plt.legend(['Actual', 'Forecasted'])
    # plt.show()

    # # Display
    # plt.figure(figsize=(20, 10))
    # plt.xlabel('Datetime')
    # plt.ylabel('Daily Prices ($)')
    # plt.title('Closing Prices from {} to {}'.format(start, end))
    # plt.plot(data)
    # plt.legend([stock_symbol] + commodity_symbols)
    # plt.show()

    # print(data.isnull().sum())

    # print(data.corr())

    # if p-value of this test is >0.05 then data is non-stationary which has to be made stationary for ARIMA to work
    # opt_diff = float('-inf')
    # for symbol in [stock_symbols[i]] + commodity_symbols:
    #     diff = 0
    #     while True:
    #         if diff == 0:
    #             adfuller_result = adfuller(data[symbol])
    #         else:
    #             adfuller_result = adfuller(data[symbol].diff(periods=diff)[diff:])
    #
    #         pvalue = adfuller_result[1]
    #         if pvalue < .05:
    #             break
    #         else:
    #             diff += 1
    #     opt_diff = max(diff, opt_diff)
    #
    # causality test
    # causality_data.append([])
    # for index in range(len(symbols) - 1):
    #     # print("Causality for {0} and {1}".format(stock_symbol, symbols[index + 1]))
    #     try:
    #         causality = grangercausalitytests(data.iloc[:, [0, index + 1]], maxlag=15, verbose=False)
    #         ssr_ftest_pvalue = causality[n_lags][0]['ssr_ftest'][1]
    #         causality_data[i].append(ssr_ftest_pvalue)
    #     except ValueError:
    #         causality_data[i].append('NaN')
    #         continue

    #
    # split into train and test
    # train_data = data.iloc[:-n_periods, :]
    # test_data = data.iloc[-n_periods:]
    #
    # # finding optimal lags
    # model = VAR(train_data.diff(periods=opt_diff)[opt_diff:])
    # sorted_model = model.select_order(maxlags=15)
    # p = np.argmin(sorted_model)
    #
    # # train the model
    # var_model = VARMAX(train_data, order=(p, 1), enforce_stationarity=True)
    # fitted_model = var_model.fit(disp=False)
    #
    # predict = fitted_model.get_prediction(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    # predictions = predict.predicted_mean
    #
    # plt.plot(test_data.index, test_data[stock_symbols[i]])
    # plt.plot(test_data.index, predictions[stock_symbols[i]])
    # plt.show()
    #
    # break

# forecasted = np.array(forecasted).T
# df = pd.DataFrame(forecasted, index=master_data[-n_periods:].index, columns=stock_symbols)
# df.to_csv('./predictions.csv')

# fig = plt.figure(figsize=(10, 7))
#
# plt.boxplot(mses)
#
# plt.show()

data = pd.read_csv('./Causality.csv', header=0, delimiter='\t')

plt.figure(figsize=(10,10))
heat_map = sns.heatmap(data, linewidth = 1 , annot = True)
plt.title("Granger-Causality Test")
plt.show()