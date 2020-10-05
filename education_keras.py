import pymongo
from keras.models import Sequential # Sequential - последовательный
from keras.layers import Dense # Dense - полносвязанный слой
import numpy as np

from download_statistics import generate_dataset_from_db


def educate_keras(datasets):
    # Создаём модель!
    model = Sequential()

    input_dim = datasets[0].shape[1]-1 # кол-во слоев на входе
    model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
    # model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, input_dim=input_dim, activation='relu'))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    for dataset in datasets:
        dataset_y = dataset[:, dataset.shape[1] - 1]
        dataset_x = dataset[:, 0:dataset.shape[1] - 1]
        model.fit(dataset_x, dataset_y, epochs=280, batch_size=1, verbose=1)
    return model

def predict(model, dataset):
    predict = model.predict(dataset[:, 0:dataset.shape[1] - 1]).T[0]
    answ = dataset[:, dataset.shape[1] - 1]
    delta = np.abs(answ - predict)
    delta_proc_mean = np.mean(delta / answ * 100)
    delta_proc_max = np.max(delta / answ * 100)

    last_price = dataset[:, dataset.shape[1] - 2: dataset.shape[1] - 1]
    delta_answ_and_max = np.abs(answ - last_price)
    delta_answ_mean = np.mean(delta_answ_and_max / answ * 100)
    delta_answ_max = np.max(delta_answ_and_max / answ * 100)
    return delta_proc_max, delta_proc_mean, delta_answ_max, delta_answ_mean


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['CryptDB']
    ethereum = db['ethereum_m5']
    datasets = generate_dataset_from_db(ethereum, [4, 2], 35, count_datasets=2)

    model = educate_keras(datasets[0:-1])

    delta_proc_max, delta_proc_mean, delta_answ_max, delta_answ_mean = predict(model, datasets[-1])


    print(model)