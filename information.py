import time

start_time = 0
def save_infomation(output_dir, date_str, test_num, classifier_list, time_step_size, colum_list, error_list) :
    info_file_name = output_dir +'/information.txt'
    start_time = time.time()
    f = open(info_file_name, mode='wt')
    f.write('\n\n\n <<< save_information >>> ')
    f.write('\ndate_str : ' + date_str)
    f.write('\ntest_num : ' + str(test_num))
    f.write('\nclassifier_list : ' + str(classifier_list))
    f.write('\ntime_step_size : ' + str(time_step_size))
    f.write('\ncolum_list : ' + str(colum_list))
    f.write('\nerror_list : ' + str(error_list))
    f.write('\nstart_time : ' + str(start_time))
    f.close()
    return

def save_infomation_data(output_dir, db_file_name, dataframe, X, Y, error_temp, X_train, Y_train, X_test, Y_test):
    info_file_name = output_dir + '/information.txt'

    f = open(info_file_name, mode='at')

    f.write('\n\n\n <<< save_information_data >>> ')
    f.write('\n db_file_name : ' + db_file_name)
    f.write('\n dataframe.shape and size')
    f.write(str(dataframe.shape))

    f.write('\n X : ' + str(X.shape))
    f.write('\n Y : ' + str(Y.shape))

    f.write('\n error_temp : ' + str(error_temp))

    f.write('\n X_train.shape : ' + str(X_train.shape))
    f.write('\n Y_train.shape : ' + str(Y_train.shape))

    f.write('\n X_test.shape : ' + str(X_test.shape))
    f.write('\n Y_test.shape : ' + str(Y_test.shape))

    f.close()
    return


def save_information_done(output_dir, classifier_name):
    info_file_name = output_dir + '/information.txt'
    f = open(info_file_name, mode='at')
    f.write('\n\n\n <<< save_information_done >>> ')
    end_date_str = time.strftime("%m%d_%H%M")
    f.write('\n classifier_name : ' + classifier_name)
    f.write('\n end_date_str : ' + end_date_str)
    duration = time.time() - start_time
    f.write('\n duration : ' + str(duration))
    f.close()