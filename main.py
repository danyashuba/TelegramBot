from io import BytesIO
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    mean_absolute_percentage_error
import statsmodels.api as sm
from sklearn import preprocessing
from pmdarima.arima import auto_arima
import telebot
from bot import get_historical_data
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotev
import os

load_dotev()
TOKEN = os.getenv('TOKEN')
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start(message):
    mess = f'Welcome, <b>{message.from_user.first_name}</b>! What would you like to do?'
    bot.send_message(message.chat.id, mess, parse_mode='html')


@bot.message_handler(commands=['BTC_price'])
def BTC_price(message):
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin',
        'vs_currencies': 'usd'
    }
    response = requests.get(url, params=params)
    data = response.json()
    bitcoin_price = data["bitcoin"]["usd"]
    mess = f"The current value of <b>BTC</b> is <b>{bitcoin_price}</b>"
    bot.send_message(message.chat.id, mess, parse_mode='html')


@bot.message_handler(commands=['get_chart'])
def get_chart(message):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)  # 5 years

    timestamps, prices = get_historical_data(start_date, end_date)

    # Convert timestamps to datetime objects
    dates = [datetime.fromtimestamp(timestamp / 1000) for timestamp in timestamps]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, label="Bitcoin Price (USD)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Over the Last 5 Years")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    bot.send_photo(message.chat.id, buffer)


@bot.message_handler(commands=['arima_predict'])
def arima_predict(message):
    stock_data = pd.read_csv('C:/Users/User/Desktop/BTC-USD.csv')
    stock_data = stock_data.iloc[::-1]

    stock_data = stock_data.drop(columns='Adj Close', axis=1)
    stock_data.index = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
    stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
    stock_data.head()

    x = stock_data.average.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
    stock_data.average = x_scaled

    n_test = int(0.3 * len(stock_data))
    test = stock_data[0:n_test]
    train = stock_data[n_test:]
    y = stock_data.average[::-1]
    for el in range(1, 3):
        stepwise_model = auto_arima(y, start_p=3, start_q=1,
                                    max_p=4, max_q=5, m=12,
                                    start_P=0, seasonal=False,
                                    d=el, D=0, trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
    model = sm.tsa.statespace.SARIMAX(y,
                                      order=(4, 2, 2),
                                      seasonal_order=(0, 0, 0, 12),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    results_ar = model.fit()
    pred_ar = results_ar.get_prediction(start=len(train), dynamic=False)
    pred_ci = pred_ar.conf_int()
    plt.figure(figsize=(16, 8))
    ax = train.average.plot(label='train', color='green')
    ax = test.average.plot(label='test', color='blue')
    pred_ar.predicted_mean.plot(ax=ax, label=f'ARIMA', alpha=0.7, color='red')

    ax.set_xlabel('Time check')
    ax.set_ylabel('Share price')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    bot.send_photo(message.chat.id, buffer)
    y_forecasted = pred_ar.predicted_mean
    plt.figure(figsize=(16, 8))
    mse = mean_squared_error(test.average, y_forecasted)
    mae = mean_absolute_error(test.average, y_forecasted)
    r2 = r2_score(test.average, y_forecasted)
    mslge = mean_squared_log_error(test.average, y_forecasted)
    mape = mean_absolute_percentage_error(test.average, y_forecasted)
    mess = (f"How exact the prediction is: "
            f"\nMSE = {mse}"
            f"\nMAE = {mae}"
            f"\nR2 = {r2}"
            f"\nMSLGE = {mslge}"
            f"\nMAPE = {mape}")
    bot.send_message(message.chat.id, mess, parse_mode='html')


@bot.message_handler(commands=['moving_average'])
def moving_average(message):

    def set_plot(title='', axis='', y=''):
        plt.title(title), plt.xlabel(axis), plt.ylabel(y)
        plt.legend()
        plt.show()
        return None

    stock_data = pd.read_csv('C:/Users/User/Desktop/BTC-USD.csv', sep=',')
    # print(stock_data)
    split = 0.3
    stock_data.index = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
    stock_data = stock_data.iloc[::-1]
    # print(stock_data)
    # print(stock_data.columns)
    stock_data = stock_data.drop(columns='Adj Close', axis=1)
    # print(stock_data.columns)

    stock_data["average"] = (stock_data["High"] + stock_data["Low"]) / 2
    stock_data.head()
    # print(stock_data)
    x = stock_data.average.values
    # print(x)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
    stock_data.average = x_scaled

    n_test = int(split * len(stock_data))
    test = stock_data[:n_test]
    train = stock_data[n_test:]
    plt.plot(train.average, label='Train')
    plt.plot(test.average, label='Test')
    set_plot(title='Share prices', axis="Time", y='Price')

    # ******************************moving_average*********************************
    window = [30, 10, 5]
    plt.figure(figsize=(16, 8))
    plt.plot(train['average'], label='Train')
    plt.plot(test['average'], label='Test')

    for w in window:
        y_hat = ((stock_data['average'].iloc[::-1].rolling(w).mean()).iloc[::-1])
        plt.plot(y_hat[:n_test], label='Moving average forecast window=' + str(w))
        plt.legend(loc='best')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        bot.send_photo(message.chat.id, buffer)
        y_forecasted = y_hat
        plt.figure(figsize=(16, 8))
        mse = mean_squared_error(test.average, y_hat[:n_test])
        mae = mean_absolute_error(test.average, y_hat[:n_test])
        r2 = r2_score(test.average, y_hat[:n_test])
        mslge = mean_squared_log_error(test.average, y_hat[:n_test])
        mape = mean_absolute_percentage_error(test.average, y_hat[:n_test])
        mess = (f"How exact the prediction is: "
                f"\nMSE = {mse}"
                f"\nMAE = {mae}"
                f"\nR2 = {r2}"
                f"\nMSLGE = {mslge}"
                f"\nMAPE = {mape}")
        bot.send_message(message.chat.id, mess, parse_mode='html')


bot.polling(none_stop=True)
