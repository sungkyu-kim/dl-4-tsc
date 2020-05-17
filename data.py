import information as info
import numpy as np
from utils.utils import create_directory
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

TRAIN_RATE = 0.25

def create_dataset_time_step(data, value, Xs, Ys, size, time_steps, error_list):
    error_value = 0
    ignore_time = 0

    for i in range(time_steps, size):
        for j in error_list:

            if (value[i] == j) & (ignore_time < 0):
                index = error_list.index(value[i])
                error_value = index
                if i > time_steps:
                    ignore_time = time_steps
                    Xdata = (data[i - time_steps:i])
                    Ydata = index
                    Xs.append(Xdata)
                    Ys.append(Ydata)
            ignore_time -= 1

    if (error_value == 0) & (size >= time_steps):
        Xdata = (data[size - time_steps:size])
        Ydata = error_value
        Xs.append(Xdata)
        Ys.append(Ydata)

    '''
    if error_value == 0 :
        ignore_time = 0
        for i in range (time_steps, size) :
            if ( i > time_steps ) & ( ignore_time < 0 ) :
                ignore_time = time_steps * 2
                Xdata = data[i-time_steps:i]
                Ydata = error_value
                Xs.append(Xdata)
                Ys.append(Ydata)
            ignore_time -= 1
    '''


def create_dataset(df, time_steps, colum_list, error_list):
    Xs, Ys = [], []

    data = df[colum_list]
    value = df['errorrecode'].values
    di = df['di']

    data = np.where(data < -999, 0, data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    temp_di = di[0]
    di_count = 0
    start_num = 0
    end_num = 0
    for i in range(len(di)):
        if (temp_di != di[i]) | (i == len(di) - 1):
            end_num = i
            temp_data = data[start_num:end_num]
            temp_value = value[start_num:end_num]

            create_dataset_time_step(temp_data, temp_value, Xs, Ys, di_count, time_steps, error_list)

            temp_di = di[i]
            start_num = i
            di_count = 0
        else:
            di_count += 1

    return np.array(Xs), np.array(Ys)


def create_data(db_file_name, output_dir, output_info_dir, time_step_size, colum_list, error_list):
    if os.path.exists(output_dir):
        print(output_dir + 'Already exist')
    else:
        create_directory(output_dir)

    our_dir_file = output_dir + 'x_train.npy'
    if os.path.exists(our_dir_file):
        print(our_dir_file + 'Already exist')
        return

    # for file_name in file_list :
    dataframe = pd.read_csv(db_file_name)
    df_size = len(dataframe)

    print('db_file_name : ' + db_file_name)
    print('dataframe.shape and size')
    print(dataframe.shape)
    print(df_size)

    X, Y = create_dataset(dataframe, time_step_size, colum_list, error_list)

    print('X')
    print(X.shape)
    print('Y')
    print(Y.shape)

    error_temp = []
    error_len = len(error_list)

    for i in range(error_len):
        error_temp.append(0)

    b = len(Y)
    for i in range(b):
        j = Y[i]
        error_temp[j] += 1
    print(error_temp)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TRAIN_RATE, shuffle=True, random_state=1004)

    print('X_train.shape')
    print(X_train.shape)
    print('Y_train.shape')
    print(Y_train.shape)

    print('X_test.shape')
    print(X_test.shape)
    print('Y_test.shape')
    print(Y_test.shape)

    info.save_infomation_data(output_info_dir, db_file_name, dataframe, X, Y, error_temp, X_train, Y_train, X_test,
                              Y_test)
    np.save(output_dir + 'X_train.npy', X_train)
    np.save(output_dir + 'Y_train.npy', Y_train)
    np.save(output_dir + 'X_test.npy', X_test)
    np.save(output_dir + 'Y_test.npy', Y_test)
    return


def load_data(dataset_name, data_dir):
    X_train = np.load(data_dir + 'X_train.npy')
    Y_train = np.load(data_dir + 'Y_train.npy')
    X_test = np.load(data_dir + 'X_test.npy')
    Y_test = np.load(data_dir + 'Y_test.npy')
    datasets_dict = {}
    datasets_dict[dataset_name] = (X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    return datasets_dict

