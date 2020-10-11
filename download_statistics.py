import pymongo
import requests
from datetime import datetime
from exception import raise_exception
from approximation import approximation_line
import numpy as np
from progress.bar import IncrementalBar
import redis
import pickle

STRETCH_X = 0.4
STRETCH_Y_LINE = 100
UPLIFT = 0.5 #уйти от отрицательных чисел

STRETCH_Y_VOLUME = 0.0001

DOWNLOAD_CHUNK_SIZE = {
    "m15": 7 * 12 * 4,  # 7дней
    "m5": 7 * 12 * 4,  # 7дней
}
MILLISECONDS = {
    "m15": 60 * 1000 * 15,
    "m5": 60 * 1000 * 5,
}



def geterate_dataset_from_arr(data_arr, lines_array, waiting, step_no_waiting):
    """
    Вычисляет датасет

    :param data_arr: массив свечей
    :param lines_array: массив количества аппроксимируемых точек размерности n
    :param waiting: количество свечей, в течение которых ожмдается рост
    # :return: датасет:
    #     0..n-1  - коэф-ты лин. апроксимации закрытия свечи,
    #     n..2n-1 - коэф-ты лин. апроксимации объемов торгов,
    #     2n      - среднее значение курса за самое max(lines_array) кол-ва свечей
    #     2n+1    - последнее значение курса
    """
    max_line = max(lines_array)
    dataset_len = len(data_arr) - waiting - max_line
    # dataset_width = len(lines_array) * 2 + 3
    # dataset_width = len(lines_array) + 3
    dataset_width = len(lines_array) + 4
    data_set = np.empty((dataset_len, dataset_width), dtype=np.float)
    for i in range(max_line, len(data_arr) - waiting):
        data_row = []
        for line in lines_array:
            data_row.append(np.mean([float(data_arr[j].get('close')) for j in range(i - line, i)]))
        #     y_close = [float(data_arr[j].get('close')) * STRETCH_Y_LINE for j in range(i - line, i)]
        #     x = [i * STRETCH_X for i in range(0, line)]
        #     solve = approximation_line(np.array([x, y_close]))
        #     data_row.append(solve[0] + UPLIFT)
        # for line in lines_array:
        #     y_volume = [float(data_arr[j].get('volume')) * STRETCH_Y_VOLUME for j in range(i - line, i)]
        #     x = [i * STRETCH_X for i in range(0, line)]
        #     solve = approximation_line(np.array([x, y_volume]))
        #     data_row.append(solve[0])
        # data_row.append(np.mean([float(data_arr[j].get('close')) for j in range(i - max_line, i)]))
        data_row.append(float(data_arr[i].get('close')))
        future_arr = np.array([float(data_arr[j].get('close')) for j in range(i + step_no_waiting, i + waiting)])
        data_row.append(np.max(future_arr))
        data_row.append(np.mean(future_arr))
        data_row.append(np.min(future_arr))
        data_set[i - max_line] = data_row
    return data_set


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
    return geterate_dataset_from_arr(coin, lines_array, waiting)


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


def generate_dataset_from_db(collection, lines_array, waiting, count_datasets=float('inf')):
    """
    Вычисление датасетов из коллекции

    :param collection: коллекция MongoDB
    :param lines_array: массив количества аппроксимируемых точек
    :param waiting: количество свечей, в течение которых ожмдается рост
    :param count_datasets: количество датасетов
    :return: массив датасетов
    """
    PAGE_LEN = 500

    bar = IncrementalBar("Render dataset from db", max=collection.find({}).count() // PAGE_LEN + 1)

    last_row = 0
    max_line = max(lines_array)
    package = [i for i in range(-1, max_line + waiting)]
    datasets_arr = []
    count = 0
    while len(package) > max_line + waiting and count < count_datasets:
        count += 1
        bar.next()
        package = [i for i in collection.find({}).skip(last_row).limit(PAGE_LEN)]
        package_set = geterate_dataset_from_arr(package, lines_array, waiting)
        if len(package_set):
            datasets_arr.append(package_set)
        last_row += len(package) - waiting - max_line
    bar.finish()
    return datasets_arr

def generate_wide_dataset_from_db(collection, lines_array, waiting, step_no_waiting, max_count_el=10):

    redis_key = 'generate_wide_dataset_from_db'
    r = redis.Redis()
    if r.get(redis_key):
        return pickle.loads(r.get(redis_key))


    count_delta_packages_max = {}
    count_delta_packages_min = {}
    ROUND_LEN = 2
    PAGE_LEN = 500
    last_row = 0
    max_line = max(lines_array)
    package = [i for i in range(-1, max_line + waiting)]
    bar = IncrementalBar("Render dataset from db", max=collection.find({}).count() // PAGE_LEN + 1)
    datasets_arr = []
    count = 0
    while len(package) > max_line + waiting:
        count += 1
        bar.next()
        package = [i for i in collection.find({}).skip(last_row).limit(PAGE_LEN)]
        package_dataset = geterate_dataset_from_arr(package, lines_array, waiting, step_no_waiting)
        for row in package_dataset:
            last_value = row[-4]
            max_v = row[-3]
            min_v = row[-1]
            percent_max = (max_v - last_value) / last_value * 100
            percent_min = (min_v - last_value) / last_value * 100
            i_max = round(percent_max, ROUND_LEN)
            i_min = round(percent_min, ROUND_LEN)
            add = True
            if i_max in count_delta_packages_max:
                if count_delta_packages_max[i_max] < max_count_el:
                    count_delta_packages_max[i_max] += 1
                    datasets_arr.append(row)
                    add = False
            else:
                count_delta_packages_max[i_max] = 1
            if i_min in count_delta_packages_min:
                if count_delta_packages_min[i_min] < max_count_el:
                    count_delta_packages_min[i_min] += 1
                    if add:
                        datasets_arr.append(row)
            else:
                count_delta_packages_min[i_min] = 1
        last_row += len(package) - waiting - max_line
    bar.finish()
    dp_keys_max = list(count_delta_packages_max.keys())
    dp_keys_min = list(count_delta_packages_min.keys())
    dp_keys_max.sort()
    dp_keys_min.sort()
    max_percent = max(dp_keys_max)
    min_percent = min(dp_keys_min)
    datasets_arr = np.array(datasets_arr)
    r.set(redis_key, pickle.dumps(datasets_arr))
    return datasets_arr


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['CryptDB']
    ethereum = db['ethereum_m5']
    api_params = {
         "exchange": "binance",
         "interval": "m5",
         "baseId": "ethereum",
         "quoteId": "bitcoin"
    }

    #download_coin_statistics(ethereum, api_params, datetime(2020, 1, 1))

    # datasets = generate_dataset_from_db(ethereum, [30, 15, 5, 3, 2], 35)

    datasets = generate_wide_dataset_from_db(ethereum, [30, 15, 5, 3, 2], 35, 5, 30)

    print(':)')
