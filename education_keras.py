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
    model.add(Dense(1, activation='sigmoid'))
    # на выходе при бинарной классификации, функцию активации чаще всего используют sigmoid , реже softmax
    # Компилирование модели. binary_crossentropy - опять же не случайно, а т.к. у нас два класса.
    # Метрика accuracy используется практически для всех задач классификации
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Наконец дошли до обучения модели, X и Y - понятно,
    # epoch - максимальное количество эпох до остановки
    # batch_size - сколько объектов будет загружаться за итерацию
    model.fit(dataset_x, dataset_y, epochs=15, batch_size=10, verbose=2)
    return model

if __name__ == '__main__':
    dataset = generate_dataset_from_api(
        {
            "exchange": "binance",
            "interval": "m5",
            "baseId": "ethereum",
            "quoteId": "bitcoin"
        },
        [30, 15, 5, 2],
        35
    )
    dataset_y = dataset[:, dataset.shape[1]-1]
    dataset_x = dataset[:, 0:dataset.shape[1]-1]
    model = educate_keras(dataset_x, dataset_y)
    a = dataset[10]
    a_predict = model.predict(np.array([dataset[10][:-1]]))

    predict = model.predict(dataset_x).T[0]
    delta = np.abs(dataset_y - predict)
    delta_proc = np.mean(delta/predict*100)

    loss = model.evaluate(dataset_x, dataset_y, batch_size=10)



    print(model)