import copy
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from utils import open_json

class Custom_Time:
    def __init__(self, string_4):
        self.hour = int(string_4[:2])
        self.min = int(string_4[3:5])

    def __add__(self, add_mins): 
        self.min += add_mins
        if self.min>=60:
            self.hour+=self.min//60
            self.min=self.min%60
        if self.hour>=48:
            print("Time Warning")
        if self.hour>=24:
            self.hour=self.hour%24
        return self
    
    def minus(self, other_time): 
        self.hour -= other_time.hour
        self.min -= other_time.min
        self.check_time()
        return self
    
    def check_time(self):
        if self.min>=60:
            self.hour+=self.min//60
            self.min=self.min%60
        if self.hour>=48:
            print("Time Warning")
        if self.hour>=24:
            self.hour=self.hour%24
        if self.min<0:
            self.hour-=1
            self.min+=60
        if self.hour <= -24:
            self.hour+=24
        if self.hour <= 0:
            self.hour =self.hour*-1


    def __eq__(self, other):
        return str(self) == str(other)


    def __str__(self) -> str:

        if self.hour<10:

            if self.min<10:
                return "0"+ str(self.hour) + ":0" + str(self.min)
            else:
                return "0"+ str(self.hour) + ":" + str(self.min)

        else:

            if self.min<10:
                return str(self.hour) + ":0" + str(self.min)
            else:
                return str(self.hour) + ":" + str(self.min)


def read_all_data(date):

    path_df = os.path.join(PATH_ROOT, "data", "raw", date+".csv")
    path_y = os.path.join(PATH_ROOT, "data", "raw", date+"_Stoat.csv")

    df = pd.read_csv(path_df, sep=",", encoding='unicode_escape', dtype='unicode').dropna(axis=1, how="all")
    # Remove Columns that contains nothing
    if df.shape[1] % 10 != 0 or df.shape[1]//10 != 4:
        print("Reading Data Warning, dataframe columns = " + str(df.shape[1]) + ". But it should be 40!")
    # Warn the user if the csv file is not exactly 40 columns
    df_list = []
    df_info_list = []
    for i in range(df.shape[1]//10):
        df_one_sensor = df.iloc[: , i * 10 + 1 : (i + 1) * 10].copy().dropna()
        df_one_sensor.columns = COLUMNS
        df_one_sensor = df_one_sensor.astype({COLUMNS[1]: 'float64', COLUMNS[2]: 'float64', COLUMNS[3]: 'float64', COLUMNS[4]: 'float64', COLUMNS[5]: 'float64', COLUMNS[6]: 'float64'})
        print("Size of dataframe " + str(i) + " is " + str(df_one_sensor.shape))
        df_list.append(df_one_sensor)
        df_info_list.append(df.columns[i * 10])
        df_info_list.append(df.iloc[0,i*10])
        
    df_y = pd.read_csv(path_y, sep=",", header=None, encoding='unicode_escape')
    if df_y.shape[1] != 1:
        print("Reading Data Label Warning, dataframe columns = " + str(df.shape[1]) + ". But it should be 1!")
    y_list = df_y[df_y.columns[0]].to_list()
    return [df_list, df_info_list, y_list]

def sliding_window_and_get_y_c(raw_X_data, out, window_size):
    X_data = []
    for index in range(len(raw_X_data) - window_size - 1): 
        X_data.append(raw_X_data[index: index + window_size])

    y_data = [0]*len(X_data)

    # Check time in X_data to construct y
    for ind in range(len(X_data)):
        for time in X_data[ind][:,0:1][0]:
            t_ind = time.index("T")
            # Assume there is only one T in the real-time recording column
            hour_min = time[t_ind + 1 : -3]
            for o in out:
                if o == hour_min or "0"+o == hour_min:
                    #print(time, o)
                    y_data[ind] = 1
        # Remove time from X
        X_data[ind] = X_data[ind][:,1:].astype(np.float64)
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data_c(X_train, X_valid, X_test, feature_range=(0, 1), standard = False):
    """
    Normalize data based on train set
    """
    if standard:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range = feature_range)

    for col_ind in range(X_train.shape[2]):
    
        X_train[:, :, col_ind] = scaler.fit_transform(X_train[:, :, col_ind].reshape(-1, 1)).reshape(X_train[:, :, col_ind].shape)
        X_test[:, :, col_ind] = scaler.transform(X_test[:, :, col_ind].reshape(-1, 1)).reshape(X_test[:, :, col_ind].shape)
        X_valid[:, :, col_ind] = scaler.transform(X_valid[:, :, col_ind].reshape(-1, 1)).reshape(X_valid[:, :, col_ind].shape)

    return X_train, X_test, X_valid

def normalize_data_r(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, feature_range=(0, 1), standard = False):
    """
    Normalize data based on train set
    """
    scalers = []

    for col_ind in range(X_train.shape[2]):

        if standard:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler(feature_range = feature_range)

        X_train[:, :, col_ind] = scaler.fit_transform(X_train[:, :, col_ind].reshape(-1, 1)).reshape(X_train[:, :, col_ind].shape)
        X_valid[:, :, col_ind] = scaler.transform(X_valid[:, :, col_ind].reshape(-1, 1)).reshape(X_valid[:, :, col_ind].shape)
        X_test[:, :, col_ind] = scaler.transform(X_test[:, :, col_ind].reshape(-1, 1)).reshape(X_test[:, :, col_ind].shape)
        scalers.append(scaler)
        # Assume that the main time series (the one the model predicts) is always the first column (0)

        if col_ind == 1:
            Y_train = scaler.transform(Y_train.reshape(-1, 1))
            Y_valid = scaler.transform(Y_valid.reshape(-1, 1))
            Y_test = scaler.transform(Y_test.reshape(-1, 1))

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, scalers

def normalize_data_with_scalers_r(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, scalers):
    
    for col_ind in range(X_train.shape[2]):
        scaler = scalers[col_ind]
        X_train[:, :, col_ind] = scaler.transform(X_train[:, :, col_ind].reshape(-1, 1)).reshape(X_train[:, :, col_ind].shape)
        X_valid[:, :, col_ind] = scaler.transform(X_valid[:, :, col_ind].reshape(-1, 1)).reshape(X_valid[:, :, col_ind].shape)
        X_test[:, :, col_ind] = scaler.transform(X_test[:, :, col_ind].reshape(-1, 1)).reshape(X_test[:, :, col_ind].shape)
        scalers.append(scaler)
        # Assume that the main time series (the one the model predicts) is always the first column (0)

        if col_ind == 1:
            Y_train = scaler.transform(Y_train.reshape(-1, 1))
            Y_valid = scaler.transform(Y_valid.reshape(-1, 1))
            Y_test = scaler.transform(Y_test.reshape(-1, 1))

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    
def get_raw_x_c(df_list, y_list, use_diff, is_control):
    if is_control:
        if df_list[2].shape[0] >= df_list[3].shape[0]:
            df_in = df_list[2]
        else:
            df_in = df_list[3]
    else:
        if df_list[0].shape[0] >= df_list[1].shape[0]:
            df_in = df_list[0]
        else:
            df_in = df_list[1]
        # Select the sensor data which holds smaller missing values
    #df_in.head(3)
    out = y_list[1:]
    df_in = df_in.iloc[:, 0:7]
    df_in = df_in.drop(df_in.columns[2], axis = 1)
    # Drop the third columns
    if use_diff:
        df_s = df_in.iloc[:, 1:].apply(pd.to_numeric).diff()
        df_in = pd.concat([df_in.iloc[1:, 0:1], df_s.iloc[1:, :]], axis=1)
        df_in = df_in.dropna(axis=1, how="all")
    #df_in.head(3)
    # Remove Time/h and Time/s column

    nan_values = df_in.isnull().sum().sum()
    if nan_values != 0 and is_control == False:
        print("Warning, there are still " + str(nan_values) + " NaN values in the data " )
    elif nan_values != 0 and is_control == True:
         print("Warning, there are still " + str(nan_values) + " NaN values in the control group data ")
    raw_X_data = df_in.to_numpy()

    return out, raw_X_data

def get_raw_x_and_y_r(df_list, use_diff, is_control = True , data_ind = None, moving_average = 0, moving_average_direction = "center"):
    if data_ind is not None:
        df_in = df_list[data_ind].copy()
        
    elif is_control == True and data_ind is None:
        if df_list[2].shape[0] >= df_list[3].shape[0]:
            df_in = df_list[2].copy()
        else:
            df_in = df_list[3].copy()
    elif is_control == False and data_ind is None:
        if df_list[0].shape[0] >= df_list[1].shape[0]:
            df_in = df_list[0].copy()
        else:
            df_in = df_list[1].copy()
        # Select the sensor data which holds smaller missing values
    #df_in.head(3)
    if moving_average != 0 :
        if moving_average_direction == "center":
            df_in["Resistance"] = df_in["Resistance"].rolling(int(moving_average), center = True).mean()
        elif moving_average_direction == "before":
            df_in["Resistance"] = df_in["Resistance"].rolling(int(moving_average)).mean()
        elif moving_average_direction == "after":
            df_in["Resistance"] = df_in["Resistance"].expanding(int(moving_average)).mean()
        df_in.dropna(inplace=True)

    df_in = df_in.iloc[:, 0:7]
    df_in = df_in.drop(df_in.columns[2], axis = 1)
    # Drop the third column
    df_in = df_in.drop(df_in.columns[0], axis = 1)
    # Drop the first column, time 
    if use_diff:
        df_s = df_in.iloc[:, 1:].apply(pd.to_numeric).diff()
        df_in = pd.concat([df_in.iloc[1:, 0:1], df_s.iloc[1:, :]], axis=1)
        df_in = df_in.dropna(axis=1, how="all")
    #df_in.head(3)
    # Remove Time/h and Time/s column

    nan_values = df_in.isnull().sum().sum()
    if nan_values != 0 and is_control == False:
        print("Warning, there are still " + str(nan_values) + " NaN values in the data " )
    elif nan_values != 0 and is_control == True:
         print("Warning, there are still " + str(nan_values) + " NaN values in the control group data ")
    raw_X_data = df_in.to_numpy()
    raw_Y_data = df_in["Resistance"].to_numpy()
    return raw_X_data, raw_Y_data

def preprocess_one_day_data_c(date_index, window_size, valid_per, test_per, include_control, use_diff):
    df_list = DATA_DICT[DATES[date_index]][0]
    y_list = DATA_DICT[DATES[date_index]][2]

    out, raw_X_data = get_raw_x_c(df_list, y_list, use_diff, False)
    
    X_data, y_data = sliding_window_and_get_y_c(raw_X_data, out, window_size)

    valid_ind = int(sum(y_data)*(1-test_per-valid_per))
    test_ind = int(sum(y_data)*(1-test_per))

    if include_control:
        _, raw_X_data_control = get_raw_x_c(df_list, y_list, use_diff, True)
        X_pres = X_data[y_data.astype(bool), : , :] 
        # print(X_pres.shape[0],sum(y_data)) # The first two numbers should be identicial
        # Read control group data
        X_data_control, y_data_control = sliding_window_and_get_y_c(raw_X_data_control, out, window_size)
        X_pres_control = X_data_control[y_data_control.astype(bool), : , :]
        # print(X_pres_control.shape[0],sum(y_data_control)) 
        # print(len(out) * 60, sum(y_data), y_data.shape)
        # The first two numbers should be close, the third number is the size of the data
        
        valid_ind_control = int(sum(y_data_control)*(1-test_per-valid_per))
        test_ind_control = int(sum(y_data_control)*(1-test_per))

        X_train = np.concatenate((X_pres[:valid_ind,:,:],X_pres_control[:valid_ind_control,:,:]))
        X_valid = np.concatenate((X_pres[valid_ind:test_ind,:,:],X_pres_control[valid_ind_control:test_ind_control,:,:]))
        X_test = np.concatenate((X_pres[test_ind:,:,:],X_pres_control[test_ind_control:,:,:]))
        y_train = np.concatenate((np.ones((valid_ind, 1)),np.zeros((valid_ind_control, 1))))
        y_valid = np.concatenate((np.ones((test_ind-valid_ind, 1)),np.zeros((test_ind_control - valid_ind_control, 1))))
        y_test = np.concatenate((np.ones((sum(y_data)-test_ind, 1)),np.zeros((sum(y_data_control)-test_ind_control, 1))))

    else:
        X_train = X_data[:valid_ind]
        X_valid = X_data[valid_ind:test_ind]
        X_test = X_data[test_ind:]
        y_train = y_data[:valid_ind]
        y_valid = y_data[valid_ind:test_ind]
        y_test = y_data[test_ind:]

        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=SEED)#random_state=seed

    X_train, X_test, X_valid = normalize_data_c(X_train, X_valid, X_test, (0, 1), False)

    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_valid = torch.from_numpy(X_valid).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor).reshape(X_train.shape[0], 1)
    y_valid  = torch.from_numpy(y_valid).type(torch.Tensor).reshape(X_valid.shape[0], 1)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).reshape(X_test.shape[0], 1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_data, y_data

def remove_one_column_from_tensor(t, col_ind):
    # Must be a 3d tensor
    return torch.cat([t[:, :, :col_ind], t[:, :, col_ind+1:]], dim=2)

def remove_resistance_from_x(X_train, X_valid, X_test):
    X_train = remove_one_column_from_tensor(X_train, 1)
    X_valid = remove_one_column_from_tensor(X_valid, 1)
    X_test = remove_one_column_from_tensor(X_test, 1)

    return X_train, X_valid, X_test

def preprocess_one_day_data_r(data_dict, date_index, window_size, valid_per, test_per, 
                              remove_resist = True, is_random_shuffle = True, use_diff = False, 
                              is_control = True, scalers = None, is_standard_scaler = False, 
                              data_ind = None, moving_average = 0, moving_average_direction = "center", 
                              seed = 1234):
    
    df_list = data_dict[DATES[date_index]][0]
    #y_list = data_dict[DATES[date_index]][2]

    raw_X_data, raw_y_data = get_raw_x_and_y_r(df_list, use_diff, is_control, data_ind, moving_average, moving_average_direction)
    
    X_data, y_data = [], []
    for index in range(len(raw_X_data) - window_size - 1): 
        X_data.append(raw_X_data[index: index + window_size])
        y_data.append(raw_y_data[index + window_size])

    X_data = np.array(X_data).astype(np.float64)
    y_data = np.array(y_data).astype(np.float64)
    if is_random_shuffle:
        #X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=int(test_per*len(raw_X_data)/100), random_state=seed)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=100, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=int(valid_per*len(raw_X_data)), random_state=seed)

    else:
        valid_ind = int(len(raw_X_data)*(1-test_per-valid_per))
        test_ind = int(len(raw_X_data)*(1-test_per))
        
        X_train = X_data[:valid_ind]
        X_valid = X_data[valid_ind:test_ind]
        X_test = X_data[test_ind:]
        y_train = y_data[:valid_ind]
        y_valid = y_data[valid_ind:test_ind]
        y_test = y_data[test_ind:]

    if scalers is None:
        X_train, X_valid, X_test, y_train, y_valid, y_test, scalers_out = normalize_data_r(X_train, X_valid, X_test, y_train, y_valid, y_test, feature_range=(0, 1), standard = is_standard_scaler)
    else: 
        X_train, X_valid, X_test, y_train, y_valid, y_test = normalize_data_with_scalers_r(X_train, X_valid, X_test, y_train, y_valid, y_test, scalers)
        scalers_out = scalers

    X_train = torch.from_numpy(X_train.astype(np.float64)).type(torch.Tensor)
    Y_train = torch.from_numpy(y_train.astype(np.float64)).type(torch.Tensor)
    X_valid = torch.from_numpy(X_valid.astype(np.float64)).type(torch.Tensor)
    Y_valid = torch.from_numpy(y_valid.astype(np.float64)).type(torch.Tensor)
    X_test = torch.from_numpy(X_test.astype(np.float64)).type(torch.Tensor)
    Y_test = torch.from_numpy(y_test.astype(np.float64)).type(torch.Tensor)

    if remove_resist:
        X_train, X_valid, X_test = remove_resistance_from_x(X_train, X_valid, X_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, X_data, y_data, scalers_out


def get_stoat_events(date_index, window_size, use_diff = False):
    df_list = DATA_DICT[DATES[date_index]][0]
    y_list = DATA_DICT[DATES[date_index]][2]

    out, raw_X_data = get_raw_x_c(df_list, y_list, use_diff, False)
    
    X_data, y_data = sliding_window_and_get_y_c(raw_X_data, out, window_size)

    return X_data, y_data

# code for average

def add_suffix_to_col_names(column_names):
    new_cols = []
    i = 0
    for c in copy.deepcopy(column_names):
        if i !=0:
            c+="_1"
        i+=1
        new_cols.append(c)
    return new_cols

def calculate_average_resistance(row):
    #print(row['Resistance_1'], row['Resistance'])
    if np.isnan(row['Resistance']) and np.isnan(row['Resistance_1']) == False:
      return row['Resistance_1']
    elif np.isnan(row['Resistance']) == False and np.isnan(row['Resistance_1']):
      return row['Resistance']
    elif np.isnan(row['Resistance']) and np.isnan(row['Resistance_1']):
      return row['Resistance']
    elif np.isnan(row['Resistance']) == False and np.isnan(row['Resistance'])== False:
      return (row['Resistance'] + row['Resistance_1'])/2
    print("F")

def average_resistance_for_one_group(date, a_ind, b_ind):
    df_a = DATA_DICT[date][0][a_ind].copy()
    df_b = DATA_DICT[date][0][b_ind].copy()
    cols = add_suffix_to_col_names(df_a.columns)
    df_b.columns = cols
    df_merged = pd.merge(left=df_a, right=df_b, on=['Real-time Recording'], how='outer')
    #print(df_merged.columns)
    df_merged["AVG Resistance"] = df_merged.apply(calculate_average_resistance, axis=1)
    df_merged["Resistance"] = df_merged["AVG Resistance"]
    df_merged = df_merged.dropna(axis=0, how="any")
    df_out = df_merged.iloc[:,0:9]
    return df_out

def average_resistance_for_one_day(date):
    df_list = []
    # Experiment group
    for pairs in [[0,1], [1,0]]:
        df_avg = average_resistance_for_one_group(date, pairs[0], pairs[1])
        df_list.append(df_avg)
    # Control group
    for pairs in [[2,3], [3,2]]:
        df_avg = average_resistance_for_one_group(date, pairs[0], pairs[1])
        df_list.append(df_avg)

    df_info_list = DATA_DICT[date][1]
    y_list = DATA_DICT[date][2]

    return [df_list, df_info_list, y_list]

def get_avg_data_dict():
    avg_data_dict = {}
    for date in DATES:
        avg_data_dict[date] = average_resistance_for_one_day(date)
    return avg_data_dict

# code for percentage

def percentage_change_for_one_day(date):
    df_list = []
    # Experiment group
    for i in range(4):
        df = DATA_DICT[date][0][i].copy()
        df["Resistance"] = (df["Resistance"] - df["Resistance"][0])/df["Resistance"][0]
        df_list.append(df)

    df_info_list = DATA_DICT[date][1]
    y_list = DATA_DICT[date][2]

    return [df_list, df_info_list, y_list]

def get_per_data_dict():
    per_data_dict = {}
    for date in DATES:
        per_data_dict[date] = percentage_change_for_one_day(date)
    return per_data_dict

# main

PATH_ROOT = Path(os.getcwd()).absolute()

metadata = open_json(os.path.join(
    PATH_ROOT, "metadata.json"))

DATES = metadata["dates"]
SEED = int(metadata["seeds"][0])

WINDOW_SIZE = int(metadata["window_sizes"][0])
VALID_PER =  float(metadata["valid_per"][0])
TEST_PER = float(metadata["test_per"][0])
THRESHOLD =  float(metadata["threshold"][0])
COLUMNS = metadata["columns"]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DICT = {}
for date in DATES:
    DATA_DICT[date] = read_all_data(date)
    print(date + " data has been read")

