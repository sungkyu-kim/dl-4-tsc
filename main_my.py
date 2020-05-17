from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime

RANDOM_SEED = 42
TRAIN_RATE = 0.25
itr = "itr_8"
#VERBOSE = True
#VERBOSE = False
VERBOSE = False
DATASET_NAME = ['JapaneseVowels','RM']
DB_FILE_NAME = ['./data/JapaneseVowels/','./data/RM/UB.csv']
#DB_FILE_NAME = ['./data/JapaneseVowels/','./data/RM/tb_hass_washer_error.csv']
DATA_DIR = './data/'
RESULT_DIR = './results/'

def fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory, test_info_str, metrics_file_str, png_str):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, VERBOSE)

    classifier.fit(x_train, y_train, x_test, y_test, y_true, test_info_str, metrics_file_str, png_str)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)

def create_dataset_time_step(data, value, Xs, Ys, size, time_steps, error_list) :    
    error_value = 0
    ignore_time = 0

    for i in range(time_steps, size) :
        for j in error_list :
            
            if (value[i] == j) & (ignore_time<0) :                
                index = error_list.index(value[i])
                error_value = index
                if i > time_steps :
                    ignore_time = time_steps
                    Xdata = (data[i-time_steps:i])
                    Ydata = index
                    Xs.append(Xdata)
                    Ys.append(Ydata)
            ignore_time-=1

    if (error_value == 0) & (size >= time_steps) :
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

    data = np.where(data<-999, 0, data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)    

    temp_di = di[0]
    di_count = 0
    start_num = 0
    end_num = 0
    for i in range(len(di)):
        if (temp_di != di[i]) | (i == len(di)-1):
            end_num = i            
            temp_data = data[start_num:end_num]
            temp_value = value[start_num:end_num]

            create_dataset_time_step(temp_data, temp_value, Xs, Ys, di_count, time_steps, error_list)

            temp_di = di[i]
            start_num = i
            di_count = 0
        else :
            di_count += 1

    return np.array(Xs), np.array(Ys)

def create_data(db_file_name, output_dir, output_info_dir, time_step_size, colum_list, error_list) :
    if os.path.exists(output_dir) :
        print(output_dir + 'Already exist')
    else :
        create_directory(output_dir)

    our_dir_file = output_dir + 'x_train.npy'
    if os.path.exists(our_dir_file) :
        print(our_dir_file + 'Already exist')
        return
    
    #for file_name in file_list :
    dataframe = pd.read_csv(db_file_name)
    df_size = len(dataframe)

    print('db_file_name : ' + db_file_name)    
    print('dataframe.shape and size')    
    print(dataframe.shape)
    print(df_size)

    X,Y = create_dataset(dataframe, time_step_size, colum_list, error_list)

    print('X')
    print(X.shape)
    print('Y')
    print(Y.shape)

    error_temp = []
    error_len = len(error_list)
    
    for i in range (error_len) :
        error_temp.append(0)

    b = len(Y)
    for i in range (b) :
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

    save_infomation_data(output_info_dir, db_file_name, dataframe, X, Y, error_temp, X_train, Y_train, X_test, Y_test)
    np.save(output_dir + 'X_train.npy', X_train)
    np.save(output_dir + 'Y_train.npy', Y_train)
    np.save(output_dir + 'X_test.npy', X_test)
    np.save(output_dir + 'Y_test.npy', Y_test)
    return

def load_data(dataset_name, data_dir) :
    X_train = np.load(data_dir + 'X_train.npy')
    Y_train = np.load(data_dir + 'Y_train.npy')
    X_test = np.load(data_dir + 'X_test.npy')
    Y_test = np.load(data_dir + 'Y_test.npy')
    datasets_dict = {}
    datasets_dict[dataset_name] = (X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy())
    return datasets_dict

start_time = 0
def save_infomation(output_dir, date_str, test_num, classifier_list, time_step_size, colum_list) :    
    info_file_name = output_dir +'/information.txt'
    start_time = time.time()    
    f = open(info_file_name, mode='wt')    
    f.write('\n\n\n <<< save_infomation >>> ')
    f.write('\ndate_str : ' + date_str)
    f.write('\ntest_num : ' + str(test_num))
    f.write('\nclassifier_list : ' + str(classifier_list))
    f.write('\ntime_step_size : ' + str(time_step_size))
    f.write('\ncolum_list : ' + str(colum_list))    
    f.write('\nstart_time : ' + str(start_time))
    f.close()
    return

def save_infomation_data(output_dir, db_file_name, dataframe, X, Y, error_temp, X_train, Y_train, X_test, Y_test) :    
    info_file_name = output_dir +'/information.txt'

    f = open(info_file_name, mode='at')    
    
    f.write('\n\n\n <<< save_infomation_data >>> ')
    f.write('\n db_file_name : ' + db_file_name)    
    f.write('\n dataframe.shape and size')    
    f.write(str(dataframe.shape))    

    f.write('\n X : ' + str(X.shape))
    f.write('\n Y : ' + str(X.shape))
    
    f.write('\n error_temp : ' + str(error_temp))
    
    f.write('\n X_train.shape : ' + str(X_train.shape))
    f.write('\n Y_train.shape : ' + str(Y_train.shape))
    
    f.write('\n X_test.shape : ' + str(X_test.shape))
    f.write('\n Y_test.shape : ' + str(Y_test.shape))
    
    f.close()
    return

def save_information_done(output_dir, classifier_name) :
    info_file_name = output_dir +'/information.txt'
    f = open(info_file_name, mode='at')    
    f.write('\n\n\n <<< save_information_done >>> ')
    end_date_str = time.strftime("%m%d_%H%M")
    f.write('\n classifier_name : ' + classifier_name)
    f.write('\n end_date_str : ' + end_date_str)
    duration = time.time() - start_time
    f.write('\n duration : ' + str(duration))    
    f.close()

def main_test(test_num, classifier_list, time_step_size, colum_list, error_list, output_dir):
    np.random.seed(RANDOM_SEED)
    #tf.random.set_seed(RANDOM_SEED)
    # this is the code used to launch an experiment on a dataset
    dir_number = 0
    dataset_name = DATASET_NAME[test_num]

    date_str = time.strftime("%m%d_%H%M")
    test_str = dataset_name + "/"+ date_str
    print("test_str %s"%(test_str))
    sub_output_dir = output_dir + test_str + '/'

    if os.path.exists(sub_output_dir):
        print(sub_output_dir + 'Already done')
    else:
        create_directory(sub_output_dir)

    #png dir
    png_str_2 = sub_output_dir + str(dir_number) + '_png/'
    if os.path.exists(png_str_2):
        print(png_str_2 + 'Already done')
    else:
        create_directory(png_str_2)

    #metrics dir
    metrics_str_2 = sub_output_dir  + str(dir_number) + '_metrics/'
    if os.path.exists(metrics_str_2):
        print(metrics_str_2 + 'Already done')
    else:
        create_directory(metrics_str_2)
    metrics_file_str2 = metrics_str_2 + 'df_metrics.txt'

    save_infomation(sub_output_dir, date_str, test_num, classifier_list, time_step_size, colum_list)

    if test_num == 0 :
        data_dir = DB_FILE_NAME[test_num]        
    else :
        data_dir = sub_output_dir + str(dir_number) + "_data/"
        create_data(DB_FILE_NAME[test_num], data_dir, sub_output_dir, time_step_size, colum_list, error_list)

    dir_number += 1
    for classifier_name in classifier_list :
        print('\n\n ===== classifier_list =====')
        print('Method: ', dataset_name, classifier_name, itr)
        print('')

        metrics_str_1 = output_dir + str(dir_number) + "_" + classifier_name + '/'
        metrics_file_str1 = metrics_str_1 + 'df_metrics.txt'
        metrics_file_str = []
        metrics_file_str.append(metrics_file_str1)
        metrics_file_str.append(metrics_file_str2)
        if os.path.exists(metrics_str_1):
            print('Already done')
        else:
            create_directory(metrics_str_1)

        png_str_1 = output_dir + str(dir_number) + "_" + classifier_name + '/'
        png_str = []
        png_str.append(png_str_1)
        png_str.append(png_str_2)
        if os.path.exists(png_str_1):
            print('Already done')
        else:
            create_directory(png_str_1)

        test_dir_df_metrics = sub_output_dir + 'df_metrics.csv'
        test_str = sub_output_dir + str(dir_number) + "_" + classifier_name + '/'
        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:
            create_directory(test_str)
            #datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
            datasets_dict = load_data(dataset_name, data_dir)
            test_info_str = date_str + "_" + str(dir_number) + "_" + classifier_name
            fit_classifier(datasets_dict, dataset_name, classifier_name, test_str, test_info_str, metrics_file_str, png_str)
            print('DONE')
            create_directory(test_str + '/DONE')
            save_information_done(sub_output_dir, classifier_name)
        dir_number += 1
    return


############################################### main
TEST_NUM = 0
TIME_STEP_SIZE = 10

#classifier_list = ['fcn','mlp','resnet','cnn','mcnn','twiesn','mcdcnn','inception']
#COLUM_LIST = ["courseoption1","courseoption2","courseoption3","courseoption4","weightvalue","washing1","washing2" ,"washing3","washing4","ipmtemp","doorlock"]

#classifier_list = XXXXXXXXXX ['tlenet', 'encoder'] XXXXX

classifier_list = ['fcn','mlp','resnet','cnn','mcdcnn','inception']
COLUM_LIST = ["courseoption1","courseoption2",'ipmtemp','doorlock']
ERROR_LIST = ['NAN','UB','DC','DDC']

date_str = time.strftime("%m%d_%H%M")
test_str = DATASET_NAME[TEST_NUM]+ "/" + date_str
output_dir = RESULT_DIR + test_str + '/'

if os.path.exists(output_dir):
    print(output_dir + 'Already done')
else:
    create_directory(output_dir)

main_test(TEST_NUM, classifier_list, TIME_STEP_SIZE, COLUM_LIST, ERROR_LIST, output_dir)
main_test(TEST_NUM, classifier_list, TIME_STEP_SIZE, COLUM_LIST, ERROR_LIST, output_dir)
main_test(TEST_NUM, classifier_list, TIME_STEP_SIZE, COLUM_LIST, ERROR_LIST, output_dir)
main_test(TEST_NUM, classifier_list, TIME_STEP_SIZE, COLUM_LIST, ERROR_LIST, output_dir)
main_test(TEST_NUM, classifier_list, TIME_STEP_SIZE, COLUM_LIST, ERROR_LIST, output_dir)

