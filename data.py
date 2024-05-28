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

    path_df = os.path.join(PATH_ROOT, "data", date+".csv")
    path_y = os.path.join(PATH_ROOT, "data", date+"_Stoat.csv")

    df = pd.read_csv(path_df, sep=",", encoding='unicode_escape', dtype='unicode').dropna(axis=1, how="all")
    # Remove Columns that contains nothing
    if df.shape[1] % 10 != 0 or df.shape[1]//10 != 4:
        print("Reading Data Warning, dataframe columns = " + str(df.shape[1]) + ". But it should be 40!")
    # Warn the user if the csv file is not exactly 40 columns
    df_list = []
    df_info_list = []
    for i in range(df.shape[1]//10):
        df_one_sensor = df.iloc[: , i * 10 + 1 : (i + 1) * 10].copy().dropna()
        print("Size of dataframe " + str(i) + " is " + str(df_one_sensor.shape))
        df_list.append(df_one_sensor)
        df_info_list.append(df.columns[i * 10])
        df_info_list.append(df.iloc[0,i*10])
        
    df_y = pd.read_csv(path_y, sep=",", header=None, encoding='unicode_escape')
    if df_y.shape[1] != 1:
        print("Reading Data Label Warning, dataframe columns = " + str(df.shape[1]) + ". But it should be 1!")
    y_list = df_y[df_y.columns[0]].to_list()
    return [df_list, df_info_list, y_list]

def sliding_window_and_get_y(raw_X_data, out, window_size):
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

def normalize_data(X_train, X_valid, X_test, feature_range=(0, 1), standard = False):
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

def preprocess_one_day_data(date_index, window_size, valid_per, test_per, include_control, use_diff):
    df_list = DATA_DICT[DATES[date_index]][0]
    y_list = DATA_DICT[DATES[date_index]][2]

    
    if df_list[0].shape[0] >= df_list[1].shape[0]:
        df_in = df_list[0]
    else:
        df_in = df_list[1]
        # Select the sensor data which holds smaller missing values
    #df_in.head(3)
    out = y_list[1:]
    df_in = df_in.iloc[:, 0:7]
    if use_diff:
        df_s = df_in.iloc[:, 1:].apply(pd.to_numeric).diff()
        df_in = pd.concat([df_in.iloc[1:, 0:1], df_s.iloc[1:, :]], axis=1)
        df_in = df_in.dropna(axis=1, how="all")
    #df_in.head(3)
    # Remove Time/h and Time/s column

    nan_values = df_in.isnull().sum().sum()
    if nan_values != 0:
        print("Warning, there are still " + str(nan_values) + " NaN values in the data " + DATES[date_index])
    raw_X_data = df_in.to_numpy()
    X_data, y_data = sliding_window_and_get_y(raw_X_data, out, window_size)

    valid_ind = int(sum(y_data)*(1-test_per-valid_per))
    test_ind = int(sum(y_data)*(1-test_per))

    if include_control:
        X_pres = X_data[y_data.astype(bool), : , :] 
        # print(X_pres.shape[0],sum(y_data)) # The first two numbers should be identicial
        # Read control group data
        if df_list[2].shape[0] >= df_list[3].shape[0]:
            df_in_control = df_list[2].iloc[:, 0:7]
        else:
            df_in_control = df_list[3].iloc[:, 0:7]
            # Select the sensor data which holds smaller missing values

        if use_diff:
            df_s_control = df_in_control.iloc[:, 1:].apply(pd.to_numeric).diff()
            df_in_control = pd.concat([df_in_control.iloc[1:, ], df_s_control.iloc[1:, 0:1]], axis=1)
            df_in_control = df_in_control.dropna(axis=1, how="all")

        nan_values_control = df_in_control.isnull().sum().sum()
        if nan_values_control != 0:
            print("Warning, there are still " + str(nan_values_control) + " NaN values in the control group data " + DATES[date_index])
        raw_X_data_control = df_in_control.to_numpy()
        X_data_control, y_data_control = sliding_window_and_get_y(raw_X_data_control, out, window_size)
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

        #X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=SEED)

    X_train, X_test, X_valid = normalize_data(X_train, X_valid, X_test, (0, 1), False)

    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_valid = torch.from_numpy(X_valid).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor).reshape(X_train.shape[0], 1)
    y_valid  = torch.from_numpy(y_valid).type(torch.Tensor).reshape(X_valid.shape[0], 1)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).reshape(X_test.shape[0], 1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_data, y_data

PATH_ROOT = Path(os.getcwd()).absolute()

metadata = open_json(os.path.join(
    PATH_ROOT, "metadata.json"))

DATES = metadata["dates"]
SEED = int(metadata["seeds"][0])
COLUMNS = metadata["columns"]

WINDOW_SIZE = int(metadata["window_sizes"][0])
VALID_PER =  float(metadata["valid_per"][0])
TEST_PER = float(metadata["test_per"][0])
THRESHOLD =  float(metadata["threshold"][0])

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DICT = {}
for date in DATES:
    DATA_DICT[date] = read_all_data(date)
    print(date + " data has been read")

