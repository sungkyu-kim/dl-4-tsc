import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils.utils import create_directory

def create_graph_data(db_file_name, output_dir, colum_list, error_list) :
    dataframe = pd.read_csv(db_file_name)
    df_size = len(dataframe)
    print(dataframe.shape)
    print(df_size)

    if os.path.exists(output_dir) :
        print('Already done')
    else :
        create_directory(output_dir)
    
    i = 1
    f = open(output_dir + 'info.txt', mode='wt')
    for colum in colum_list :
        print('colum : ' + colum)
        data = dataframe[colum].values

        data = np.where(data<-999, 0, data)
        
        good_data = np.array(data[(np.where(data > -999))])
        bad_data = np.array(data[(np.where(data < -999))])

        size = len(data)
        g2 = len(good_data)
        b2 = len(bad_data)

        remain = g2/size * 100

        sns.distplot(good_data)
        info = '[' + colum + ']' + ' size : ' + str(size) + ' , good : ' + str(g2) + ' , bad : ' + str(b2) + ' , per : ' + str(int(remain)) + '% '

        plt.title(info)
        f.write(info+'\n')
        file_name = output_dir + str(i)+'_'+colum+'.png'
        plt.savefig(file_name)
        #plt.show()
        i+=1
    f.close()
    return

def main_graph(db_file_name, output_dir, colum_list, error_list) :
    date_str = time.strftime("%m%d_%H%M")

    output_dir = output_dir + date_str + '/'

    create_graph_data(db_file_name, output_dir, colum_list, error_list)
    return

########################################### main

TEST_NUM = 1
RESULT_DIR = './results/Data_Graph/'
date_str = time.strftime("%m%d_%H%M")

classifier_list = ['fcn', 'mlp', 'resnet', 'cnn', 'mcdcnn', 'inception']


DB_FILE_NAME = './data/RM/ub.csv'
COLUM_LIST = ['courseoption1','courseoption2','courseoption3','courseoption4','weightvalue','washing1','washing2','washing3','washing4','ipmtemp','doorlock']
ERROR_LIST = ['NAN', 'UB', 'HC1']

dir_name = '1_test/'
output_dir = RESULT_DIR + dir_name



main_graph(DB_FILE_NAME, output_dir, COLUM_LIST, ERROR_LIST)