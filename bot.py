import requests


def get_historical_data(start_date, end_date):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_date.timestamp(),
        "to": end_date.timestamp()
    }

    response = requests.get(url, params=params)
    data = response.json()

    timestamps = [timestamp for timestamp, _ in data["prices"]]
    prices = [price for _, price in data["prices"]]

    return timestamps, prices
