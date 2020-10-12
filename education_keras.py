import pymongo
from keras.models import Sequential  # Sequential - последовательный
from keras.layers import Dense  # Dense - полносвязанный слой
import numpy as np

from keras.models import model_from_yaml
import os

from download_statistics import generate_dataset_from_db, generate_wide_dataset_from_db

OUTPUT_DIM = 3

def educate_keras(datasets, load_from_file=False):
    # Создаём модель!

    if load_from_file and os.path.exists('model.yaml'):
        yaml_file = open('model.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights("model.h5")
        print("Loaded model from disk")
    else:
        model = Sequential()

        input_dim = datasets[0].shape[1] - OUTPUT_DIM  # кол-во слоев на входе
        model.add(Dense(OUTPUT_DIM*input_dim, input_dim=input_dim, activation='relu'))
        model.add(Dense(OUTPUT_DIM*input_dim, input_dim=OUTPUT_DIM*input_dim, activation='relu'))
        model.add(Dense(OUTPUT_DIM, input_dim=OUTPUT_DIM*input_dim, activation='relu'))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    for dataset in datasets:
        dataset_y = dataset[:, dataset.shape[1] - OUTPUT_DIM]
        dataset_x = dataset[:, 0:dataset.shape[1] - OUTPUT_DIM]
        model.fit(dataset_x, dataset_y, epochs=130, batch_size=1, verbose=1)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model

def evaluate_success(model, dataset):
    shape = dataset.shape[1]
    predict = model.predict(dataset[:, 0:shape - OUTPUT_DIM])
    answ = dataset[:, shape - OUTPUT_DIM: shape]

    last_price = dataset[:, dataset.shape[1] - OUTPUT_DIM - 1: dataset.shape[1] - OUTPUT_DIM]
    if OUTPUT_DIM > 2:
        last_price = last_price[np.newaxis]
        last_price_matrix = np.append(last_price, last_price, axis=2)
        for i in range(2, OUTPUT_DIM):
            last_price_matrix = np.append(last_price_matrix, last_price, axis=2)
        last_price = last_price_matrix[0]


    delta = np.abs(answ - predict) / answ * 100
    delta_mean = np.mean(delta, axis=0)
    delta_max = np.max(delta, axis=0)

    delta_answ = np.abs(answ - last_price) / answ * 100
    delta_answ_mean = np.mean(delta_answ, axis=0)
    delta_answ_max = np.max(delta_answ, axis=0)

    good_mean = delta_answ_mean - delta_mean
    good_max = delta_answ_max - delta_max

    return delta_max, delta_mean, delta_answ_max, delta_answ_mean


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['CryptDB']
    ethereum = db['ethereum_m5']
    # datasets = generate_dataset_from_db(ethereum, [4, 2], 35, count_datasets=2)
    dataset = generate_wide_dataset_from_db(ethereum, [30, 15, 5, 3, 2], 35, 5, 30)

    dataset_fit = dataset[0:dataset.shape[0]//2]
    dataset_predict = dataset[dataset.shape[0]//2: dataset.shape[0]]

    model = educate_keras([dataset_fit], True)

    delta_proc_max, delta_proc_mean, delta_answ_max, delta_answ_mean = evaluate_success(model, dataset_predict)

    print(model)
