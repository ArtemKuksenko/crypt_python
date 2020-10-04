import pymongo
import requests
from datetime import datetime
from exception import raise_exception
from approximation import approximation_line
import numpy as np
from progress.bar import IncrementalBar

STRETCH_X = 0.01
STRETCH_Y = 100

DOWNLOAD_CHUNK_SIZE = {
    "m15": 7 * 12 * 4  # 7дней
}
MILLISECONDS = {
    "m15": 60 * 15 * 1000
}


def generate_dataset_from_api(api_params, lines_array, waiting):
    """
    Генерация датасета из данных из CoinCap

    :param api_params: параметры запроса CoinCap
    :param lines_array: массив количества аппроксимируемых точек размерности n
    :param waiting: количество свечей, в течение которых ожмдается рост
    :return: датасет:
        0..n-1  - коэф-ты лин. апроксимации закрытия свечи,
        n..2n-1 - коэф-ты лин. апроксимации объемов торгов,
        2n      - среднее значение курса за самое max(lines_array) кол-ва свечей
        2n+1    - последнее значение курса
    """
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
    dataset_len = len(coin) - waiting - max_line
    data_set = np.empty((dataset_len, len(lines_array) * 2 + 3), dtype=np.float)
    for i in range(max_line, len(coin) - waiting):
        data_row = []
        for line in lines_array:
            y_close = [float(coin[j].get('close')) * STRETCH_Y for j in range(i - line, i)]
            x = [i * STRETCH_X for i in range(0, line)]
            solve = approximation_line(np.array([x, y_close]))
            data_row.append(solve[0])
        for line in lines_array:
            y_volume = [float(coin[j].get('volume')) * STRETCH_Y for j in range(i - line, i)]
            x = [i * STRETCH_X for i in range(0, line)]
            solve = approximation_line(np.array([x, y_volume]))
            data_row.append(solve[0])
        data_row.append(np.mean([float(coin[j].get('close')) for j in range(i - max_line, i)]))
        data_row.append(float(coin[i].get('close')))
        data_row.append(max([float(coin[j].get('close')) for j in range(i, i + waiting)]))
        data_set[i - max_line] = data_row

    return data_set


def download_coin_statistics(collection, api_params, date_start=datetime(2020, 1, 1), date_end=datetime.now()):
    """
    Загрузка статистики в коллкцию mongoDB

    :param collection: коллекция MongoDB
    :param api_params: параметры запроса CoinCap
    :param date_start: дата старта
    :param date_end:  дата окончания
    :return: отрисовывает полосу загрузки датасета
    """

    date_start = date_start.timestamp() * 1000
    date_end = date_end.timestamp() * 1000

    time_len = MILLISECONDS.get(api_params.get('interval'))
    time_step = time_len * DOWNLOAD_CHUNK_SIZE.get(api_params.get('interval'))
    time_a = date_start
    bar = IncrementalBar(api_params.get('baseId'), max=(date_end - date_start) // time_step + 1)

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
            for candle in package:
                candle["period"] = datetime.fromtimestamp(candle.get("period") // 1000)
            collection.insert_many(package)
        else:
            bar.finish()
            time_a = date_end

    return


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['CryptDB']
    ethereum = db['ethereum']
    download_coin_statistics(
        ethereum
        , {
            "exchange": "binance",
            "interval": "m15",
            "baseId": "ethereum",
            "quoteId": "bitcoin"
        },
    )
