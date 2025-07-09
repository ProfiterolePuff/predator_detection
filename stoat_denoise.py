import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import (create_dir,
                    append_result_to_csv_file,
                    check_file_existence,
                    create_empty_result_csv_file)

from model import *
from plot import *

from data import * 

def get_result_path():
    if IS_DIFF:
        path_out = os.path.join(PATH_ROOT, "results", "differencing_denoise_results.csv")
    elif IS_AVG:
        path_out = os.path.join(PATH_ROOT, "results", "averaged_denoise_results.csv")
    elif IS_PER:
        path_out = os.path.join(PATH_ROOT, "results", "percentage_denoise_results.csv")
    else:
        path_out = os.path.join(PATH_ROOT, "results", "normal_denoise_results.csv")
    return path_out

def run_denoise_experiments(data_dict, d_ind, remove_resist = True, moving_average = 0, 
                            moving_average_direction = "center", plot_detailed_results = False, 
                            plot_general_results = False, data_ind_pairs = None, 
                            is_standard_scaler = False, seed = 1234):
    #lstm_models = []
    
    path_out = get_result_path()
    #create_dir(path_out)
    print(check_file_existence(path_out))
    if check_file_existence(path_out) == False:
        # check the reslt file path, it file exists obtain new file name 
        first_result_line_list = ["date", "seed", "mse_control", "mse_experiment", "experiment_index", 
                                  "control_index", "is_diff", "is_avg", "is_per", "scaler_type", 
                                  "moving_averge_num", "moving_average_direction", "success_num", 
                                  "events_num", "success_rate", "slopes"]
        create_empty_result_csv_file(path_out, first_result_line_list)


    for data_ind_pair in data_ind_pairs:
        X_train, X_valid, X_test, y_train, y_valid, y_test, _, _, scalers = preprocess_one_day_data_r(
            data_dict, d_ind, WINDOW_SIZE, VALID_PER, TEST_PER, remove_resist = remove_resist, 
            is_random_shuffle = True, use_diff = IS_DIFF, is_control = True, scalers = None, 
            is_standard_scaler = is_standard_scaler, data_ind = data_ind_pair[1], 
            moving_average = moving_average, moving_average_direction = moving_average_direction, seed = seed)
        lstm_model, lstm_losses = grid_search_rnn(X_train, y_train, X_valid, y_valid, "LSTM", DEVICE, seed)
        #lstm_models.append(lstm_models)

        #plt.plot(lstm_losses)
        #plt.show()

        X_time, y_time = get_stoat_events(d_ind, WINDOW_SIZE)
        X_train_e, X_valid_e, X_test_e, y_train_e, y_valid_e, y_test_e, _, _ , _= preprocess_one_day_data_r(
            data_dict, d_ind, WINDOW_SIZE, VALID_PER, TEST_PER, remove_resist = remove_resist, 
            is_random_shuffle = False, use_diff = IS_DIFF, is_control = False, scalers = None, 
            is_standard_scaler = is_standard_scaler, data_ind = data_ind_pair[0], 
            moving_average = moving_average, moving_average_direction = moving_average_direction, seed = seed)
        # Different Scalers for experiments and control groups
        X_all_e = torch.cat((X_train_e, X_valid_e, X_test_e), 0)
        y_all_e = torch.cat((y_train_e, y_valid_e, y_test_e), 0)
        
        mse_fun = torch.nn.MSELoss(reduction = "mean")
        mse_c = round(mse_fun(lstm_model(X_test.to(DEVICE)).cpu(), y_test.cpu()).item(), 5)

        mse_e = round(mse_fun(lstm_model(X_all_e.to(DEVICE)).cpu(), y_all_e.cpu()).item(), 5)

        #plot_model_predictions(lstm_model(X_test.to(DEVICE)), y_test.cpu(), DATES[d_ind], data_ind_pair[1])
        if plot_general_results:
            plot_model_predictions(lstm_model(X_all_e.to(DEVICE)).cpu(), y_all_e.cpu(), DATES[d_ind], data_ind_pair[0])
            
            plot_results(lstm_model(X_all_e.to(DEVICE)), y_all_e.cpu(), y_time, DATES[d_ind], data_ind_pair)

        success_num, events_num, slopes = plot_results_small_windows(
            lstm_model(X_all_e.to(DEVICE)), y_all_e.cpu(), y_time, DATES[d_ind], data_ind_pair, plot_detailed_results)
        
        # Write results to disk
        if is_standard_scaler:
            scaler_type="Standard"
        else:
            scaler_type="MinMax"
        
        
        result = [DATES[d_ind], seed, str(mse_c), str(mse_e), str(data_ind_pair[0]), str(data_ind_pair[1]), 
                  IS_DIFF, IS_AVG, IS_PER, scaler_type, str(moving_average), str(moving_average_direction), 
                  str(success_num), str(events_num), success_num/events_num, str(slopes).replace(",", ";")]
        append_result_to_csv_file(path_out, result)
        print('Save to:', path_out)

        return success_num, events_num

SENSOR_INDS_DICT={}
SENSOR_INDS_DICT["April_8"] = [[0, 2]]
SENSOR_INDS_DICT["April_9"] = [[1, 2]]
SENSOR_INDS_DICT["April_10"] = [[1, 3]]
SENSOR_INDS_DICT["April_11"] = [[0, 3]]

#DATA_INDS = [[0, 2], [0, 3], [1, 2], [1, 3]]
#DATA_INDS = [[1, 2]]

da_inds = [0, 1, 2]
ma_dirs = ["center", "before", "after"]
ma_vals = [0, 30, 60, 120, 240, 300, 600, 900]

IS_DIFF = False
IS_AVG = True
IS_PER = False

data_dict_for_denoise = DATA_DICT

if IS_AVG:
    avg_data_dict = get_avg_data_dict()
    data_dict_for_denoise = avg_data_dict 
elif IS_PER:
    per_data_dict = get_per_data_dict()
    data_dict_for_denoise = per_data_dict

#plt.rcParams.update({"figure.figsize": (15, 6), "figure.dpi": 120}) # matplotlib plot size
for date_ind in da_inds:
    for standard in [True, False]:
        for direction in ma_dirs:
            for ma in ma_vals:

                print(DATES[date_ind], ma, standard, direction)
                run_denoise_experiments(data_dict_for_denoise, date_ind, True, ma, data_ind_pairs = SENSOR_INDS_DICT[DATES[date_ind]], is_standard_scaler = standard, moving_average_direction = direction, seed=SEED)
                #print(DATES[date_ind], ma, direction)