import pymongo
from keras.models import Sequential  # Sequential - последовательный
from keras.layers import Dense  # Dense - полносвязанный слой
import numpy as np
from tensorflow.keras import activations

from keras.models import model_from_yaml
import os
import keras

from download_statistics import generate_dataset_from_db, generate_wide_dataset_from_db

OUTPUT_DIM = 2
SCALE_COEFFICIENT = 100
UPLIFT = 0.4

def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Dense(units=hp.Int('units_input',  # Полносвязный слой с разным количеством нейронов
                                 min_value=512,  # минимальное количество нейронов - 128
                                 max_value=1024,  # максимальное количество - 1024
                                 step=32),
                    input_dim=784,
                    activation=activation_choice))
    model.add(Dense(units=hp.Int('units_hidden',
                                 min_value=128,
                                 max_value=600,
                                 step=32),
                    activation=activation_choice))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def load_model(f_name="model"):
    """
    загрузка Модели
    :param f_name: наименование
    :return:
    """
    yaml_file = open(f'{f_name}.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights(f"{f_name}.h5")
    print("Loaded model from disk")
    return model


def educate_or_load_model(datasets, load_from_file=False):
    """
    Обучение или загрузка модели
    :param datasets: массив датасетов
    :param load_from_file:
    :return:
    """

    if load_from_file and os.path.exists('model.yaml'):
        model = load_model('model')
        return model

    model = Sequential()

    input_dim = datasets[0].shape[1] - OUTPUT_DIM - 1  # отнимаем кол-во слоев на входе и сам курс
    model.add(Dense(OUTPUT_DIM * input_dim, input_dim=input_dim, activation='softplus'))
    model.add(Dense(OUTPUT_DIM * input_dim, activation='softplus'))
    model.add(Dense(OUTPUT_DIM * input_dim, activation='softplus'))
    model.add(Dense(OUTPUT_DIM * input_dim, activation='softplus'))
    model.add(Dense(OUTPUT_DIM, activation='softplus'))

    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    shape = datasets[0].shape[1]
    for dataset in datasets:
        dataset_y = dataset[:, shape - OUTPUT_DIM: shape]
        dataset_x = dataset[:, 0:dataset.shape[1] - OUTPUT_DIM - 1]
        model.fit(dataset_x, dataset_y, epochs=100, batch_size=10, verbose=1)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model


def __from_dataset_to_real_price(last_price, values):
    return last_price - (values - UPLIFT) / SCALE_COEFFICIENT


def evaluate_success(model, dataset):
    """
    Функкция высчитывающая успех.
    Успехом назовем функцию, которая оценивает на сколько процентов отклонение от предсказания
    будет отличаться отклонения курса, что позволит оценить на сколько процентов наше предсказание лучше,
    чем просто сказать "курс не изменится"
    Максимальная комиссия у Binance - 0.1%
    :param model:
    :param dataset:
    :return:
    """
    shape = dataset.shape[1]
    predict = model.predict(dataset[:, 0:shape - OUTPUT_DIM - 1])
    answ = dataset[:, shape - OUTPUT_DIM: shape]

    last_price = dataset[:, dataset.shape[1] - OUTPUT_DIM - 1: dataset.shape[1] - OUTPUT_DIM]

    answ = __from_dataset_to_real_price(last_price, answ)
    predict = __from_dataset_to_real_price(last_price, predict)

    # last_price = last_price[np.newaxis]
    # last_price_matrix = np.append(last_price, last_price, axis=2)

    real_delta = np.abs(answ - last_price) / last_price * 100  # произошедшие отклонения от последней точки
    predict_delta = np.abs(answ - predict) / last_price * 100  # отклонение от резульатата
    profit = real_delta - predict_delta

    return profit, predict_delta

def __look_max_min_datasets_values(dataset):
    """
    Расчет векторов наибольших и наименьших значений,
    за исключением вектора последнего значения
    :param dataset:
    :return:
    """
    input_dim = dataset.shape[1] - OUTPUT_DIM - 1
    max_dataset_values = []
    min_dataset_values = []
    for i in range(0, input_dim):
        max_dataset_values.append(np.max(dataset.T[i]))
        min_dataset_values.append(np.min(dataset.T[i]))
    for i in range(input_dim + 1, dataset.shape[1]):
        max_dataset_values.append(np.max(dataset.T[i]))
        min_dataset_values.append(np.min(dataset.T[i]))
    return max_dataset_values, min_dataset_values

if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['CryptDB']
    ethereum = db['ethereum_m5']

    dataset = generate_wide_dataset_from_db(ethereum, [1, 2, 3, 5, 10, 30], 20, 5, 30)

    max_dataset_values, min_dataset_values = __look_max_min_datasets_values(dataset)

    dataset_fit = dataset[0:dataset.shape[0] // 2]
    dataset_estimation = dataset[dataset.shape[0] // 2: dataset.shape[0]]

    model = educate_or_load_model([dataset_fit], load_from_file=True)

    # model = load_model()
    # model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    profit_estimation, predict_delta_est = evaluate_success(model, dataset_estimation)
    profit_fit, predict_delta_fit = evaluate_success(model, dataset_fit)

    profit_estimation_mean = np.mean(profit_estimation)
    profit_fit_mean = np.mean(profit_fit)

    predict_estimation_mean = np.mean(predict_delta_est)
    predict_fit_mean = np.mean(predict_delta_fit)

    print(model)
