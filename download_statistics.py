import requests
from exception import raise_exception
from approximation import approximation_line
import numpy as np

STRETCH_X = 0.01
STRETCH_Y = 100


def generate_dataset_from_api(api_params, lines_array, waiting):
    try:
        coin = requests.get(
            'http://api.coincap.io/v2/candles',
            params=api_params,
        )
    except Exception as err:
        raise_exception("Загрузка данных", err)
        return
    coin = coin.json()
    if coin.get("error"):
        raise_exception("Api", coin.get("error"))
        return

    coin = coin.get('data')

    max_line = max(lines_array)
    dataset_len = len(coin)-waiting - max_line
    data_set = np.empty((dataset_len, len(lines_array) + 3), dtype=np.float)
    for i in range(max_line, len(coin)-waiting):
        data_row = []
        for line in lines_array:
            y = [float(coin[j].get('close')) * STRETCH_Y for j in range(i-line, i)]
            x = [i * STRETCH_X for i in range(0, line)]
            solve = approximation_line(np.array([x, y]))
            data_row.append(solve[0])
        data_row.append(np.mean([float(coin[j].get('close')) for j in range(i-max_line, i)]))
        data_row.append(float(coin[i].get('close')))
        data_row.append(max([float(coin[j].get('close')) for j in range(i, i+waiting)]))
        data_set[i-max_line] = data_row

    return data_set


generate_dataset_from_api(
    {
        "exchange": "binance",
        "interval": "m15",
        "baseId": "ethereum",
        "quoteId": "bitcoin"
    },
    [10, 5, 3],
    10
)
