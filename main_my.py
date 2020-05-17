from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os

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

from datetime import datetime
import information as info
import numpy as np
import time
import data
import learning as learn

RANDOM_SEED = 42


#VERBOSE = True
#VERBOSE = False
VERBOSE = False
DATASET_NAME = ['JapaneseVowels','RM']
DB_FILE_NAME = ['./data/JapaneseVowels/','./data/RM/UB.csv']
#DB_FILE_NAME = ['./data/JapaneseVowels/','./data/RM/tb_hass_washer_error.csv']
DATA_DIR = './data/'
RESULT_DIR = './results/'


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

    info.save_infomation(sub_output_dir, date_str, test_num, classifier_list, time_step_size, colum_list, error_list)

    if test_num == 0 :
        data_dir = DB_FILE_NAME[test_num]        
    else :
        data_dir = sub_output_dir + str(dir_number) + "_data/"
        data.create_data(DB_FILE_NAME[test_num], data_dir, sub_output_dir, time_step_size, colum_list, error_list)

    dir_number += 1
    for classifier_name in classifier_list :
        print('\n\n ===== classifier_list =====')
        print('Method: ', dataset_name, classifier_name)
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
            datasets_dict = data.load_data(dataset_name, data_dir)
            test_info_str = date_str + "_" + str(dir_number) + "_" + classifier_name
            learn.fit_classifier(datasets_dict, dataset_name, classifier_name, test_str, test_info_str, metrics_file_str, png_str, VERBOSE)
            print('DONE')
            create_directory(test_str + '/DONE')
            info.save_information_done(sub_output_dir, classifier_name)
        dir_number += 1
    return


############################################### main
TEST_NUM = 0
TIME_STEP_SIZE = 10

#classifier_list = ['fcn','mlp','resnet','cnn','mcnn','twiesn','mcdcnn','inception']
#COLUM_LIST = ["courseoption1","courseoption2","courseoption3","courseoption4","weightvalue","washing1","washing2" ,"washing3","washing4","ipmtemp","doorlock"]

#classifier_list = XXXXXXXXXX ['tlenet', 'encoder'] XXXXX

#classifier_list = ['fcn','mlp','resnet','cnn','mcdcnn','inception']
classifier_list = ['fcn']
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

