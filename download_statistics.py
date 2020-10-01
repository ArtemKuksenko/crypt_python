import pymongo
import requests
from datetime import datetime
from exception import raise_exception
from approximation import approximation_line
import numpy as np
from progress.bar import IncrementalBar

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
    data_set = np.empty((dataset_len, len(lines_array)*2 + 3), dtype=np.float)
    for i in range(max_line, len(coin)-waiting):
        data_row = []
        for line in lines_array:
            y_close = [float(coin[j].get('close')) * STRETCH_Y for j in range(i-line, i)]
            x = [i * STRETCH_X for i in range(0, line)]
            solve = approximation_line(np.array([x, y_close]))
            data_row.append(solve[0])
        for line in lines_array:
            y_volume = [float(coin[j].get('volume')) * STRETCH_Y for j in range(i-line, i)]
            x = [i * STRETCH_X for i in range(0, line)]
            solve = approximation_line(np.array([x, y_volume]))
            data_row.append(solve[0])
        data_row.append(np.mean([float(coin[j].get('close')) for j in range(i-max_line, i)]))
        data_row.append(float(coin[i].get('close')))
        data_row.append(max([float(coin[j].get('close')) for j in range(i, i+waiting)]))
        data_set[i-max_line] = data_row

    return data_set


# generate_dataset_from_api(
#     {
#         "exchange": "binance",
#         "interval": "m15",
#         "baseId": "ethereum",
#         "quoteId": "bitcoin"
#     },
#     [10, 5, 3],
#     10
# )


# Create the client
client = pymongo.MongoClient('localhost', 27017)

# Connect to our database
db = client['CryptDB']

# Fetch our series collection
ethereum = db['ethereum']

print('kek')

DOWNLOAD_CHUNK_SIZE = {
    "m15": 7*12*4 #7дней
}
MILLISECONDS = {
    "m15": 60*15*1000
}

def download_coin_statistics(collection, api_params, date_start, date_end=datetime.now().timestamp()*1000):
    time_len = MILLISECONDS.get(api_params.get('interval'))
    time_step = time_len * DOWNLOAD_CHUNK_SIZE.get(api_params.get('interval'))
    time_a = date_start
    bar = IncrementalBar(api_params.get('baseId'), max=(date_end-date_start)//time_step+1)

    while time_a < date_end:
        time_b = time_a + time_step
        api_params["start"] = time_a
        api_params["end"] = time_b
        package = requests.get(
            'http://api.coincap.io/v2/candles',
            params=api_params,
        ).json().get('data')
        if len(package):
            bar.next()
            time_a = int(package.pop().get('period')) + time_len
            collection.insert_many(package)
        else:
            bar.finish()
            time_a = date_end

    return

download_coin_statistics(collection
                         , {
        "exchange": "binance",
        "interval": "m15",
        "baseId": "ethereum",
        "quoteId": "bitcoin"
    },
1577826000000 #2020 в миллисеках
                         )

