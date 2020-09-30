from keras.models import Sequential # Sequential - последовательный
from keras.layers import Dense # Dense - полносвязанный слой
import numpy as np

from download_statistics import generate_dataset_from_api

def educate_keras(dataset_x, dataset_y):
    # Создаём модель!
    model = Sequential()
    # Добавляем первый слой Dense, первое число 12 - это количество нейронов,
    # input_dim - количество фич на вход
    # activation -  функция активации, полулинейная функция max(x, 0)
    # именно полулинейные функции позволяют получать нелинейные результаты с минимальными затратами
    model.add(Dense(12, input_dim=dataset_x.shape[1], activation='sigmoid'))
    # добавляем второй слой с 8ю нейронами
    model.add(Dense(dataset_x.shape[1], activation='sigmoid'))
    model.add(Dense(dataset_x.shape[1], activation='sigmoid'))
    model.add(Dense(dataset_x.shape[1], activation='sigmoid'))
    model.add(Dense(dataset_x.shape[1]//2, activation='sigmoid'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # на выходе при бинарной классификации, функцию активации чаще всего используют sigmoid , реже softmax
    # Компилирование модели. binary_crossentropy - опять же не случайно, а т.к. у нас два класса.
    # Метрика accuracy используется практич11ески для всех задач классификации
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    # Наконец дошли до обучения модели, X и Y - понятно,
    # epoch - максимальное количество эпох до остановки
    # batch_size - сколько объектов будет загружаться за итерацию
    model.fit(dataset_x, dataset_y, epochs=250, batch_size=30, verbose=1)
    return model

if __name__ == '__main__':
    dataset = generate_dataset_from_api(
        {
            "exchange": "binance",
            "interval": "m5",
            "baseId": "ethereum",
            "quoteId": "bitcoin"
        },
        [30, 15, 5, 3, 2],
        35
    )

    min = dataset[:, dataset.shape[1]-1].min(axis=0)
    max_subtract_min = dataset[:, dataset.shape[1]-1].max(axis=0) - min
    dataset[:, dataset.shape[1]-1] -= min
    dataset[:, dataset.shape[1]-1] /= max_subtract_min
    dataset_y = dataset[0:dataset.shape[0]//2, dataset.shape[1]-1]
    dataset_x = dataset[0:dataset.shape[0]//2, 0:dataset.shape[1]-1]


    model = educate_keras(dataset_x, dataset_y)
    a = dataset[10]
    a_predict = model.predict(np.array([dataset[10][:-1]]))

    predict = model.predict(dataset[:-dataset.shape[0]//2, 0:dataset.shape[1]-1]).T[0]
    predict *= max_subtract_min
    predict += min
    dataset[:, dataset.shape[1] - 1] *= max_subtract_min
    dataset[:, dataset.shape[1] - 1] += min
    delta = np.abs(dataset[:-dataset.shape[0]//2, dataset.shape[1]-1] - predict)
    delta_proc_mean = np.mean(delta/predict*100)
    delta_proc_max = np.max(delta/predict*100)

    loss = model.evaluate(dataset_x, dataset_y, batch_size=10)



    print(model)