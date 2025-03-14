from som_data_struct import som_data_struct
from som_normalize import som_normalize
import matplotlib.pyplot as plt
import copy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
def read_headers(filename):
    '''Function to read the headers of the data file'''
    with open(filename, 'r') as f:
        headers = f.readline().strip().split(',')
    return headers

def read_data(filename, startcol=0, endcol=None):
    '''Function to read the data from the file'''
    return np.loadtxt(filename, delimiter=',', skiprows=1, usecols=range(startcol, endcol))

def get_start_end_indices(labels_dataframe):
    '''Function to get the start and end indices of the train/test data'''
    start_test = labels_dataframe.groupby('Participant').head(1).index.tolist()
    end_test = start_test[1:] + [len(labels_dataframe)]
    return start_test, end_test

def Equal_lengths(data_dict, max_length):
    '''Function to make the lengths of all the trials equal'''
    df = pd.DataFrame(data_dict['data'])
    df.columns = data_dict['comp_names']
    labels = pd.DataFrame(data_dict['labels'])
    labels.columns = data_dict['label_names']
    start,end = get_start_end_indices(labels)
    trials = []
    labelss = []
    for i in range(len(start)):
        trials.append(df.iloc[start[i]:end[i],:])
    labelss = labels.groupby('Participant').head(1).values.tolist()
    
    # Step 2: Pad each DataFrame with NaNs to match the maximum length
    padded_trials = []
    for trial in trials:
        # Calculate the number of rows to add
        num_missing_rows = max_length - trial.shape[0]
        if num_missing_rows > 0:
            # Create a DataFrame of NaNs with the same columns and num_missing_rows rows
            nan_rows = pd.DataFrame(np.nan, index=np.arange(num_missing_rows), columns=trial.columns)
            # Append the NaN rows to the original trial DataFrame
            padded_trial = pd.concat([trial, nan_rows], ignore_index=True)
        else:
            padded_trial = trial
        padded_trials.append(padded_trial)
        #padded_labels.append(lab)
    proper_labels = []
    for label in labelss:
        proper_labels.append(np.repeat(label, max_length))
    proper_labels = np.concatenate(proper_labels, axis = 0)
    data_dict['data'] = np.vstack(padded_trials)
    data_dict['labels'] = proper_labels
    return data_dict

def fill_missing_values(data, compnames):
    # Split the data into features and target
    data = pd.DataFrame(data)
    data.columns = compnames
    data_d = copy.deepcopy(data)
    data_d = data_d.dropna()
    data_undropped_d = data.iloc[:,0:16]
    y_undropped = data_undropped_d['HR']
    X = data_d.drop('HR', axis=1)
    y = data_d['HR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf = RF(n_estimators=80, max_features= None, max_depth= 15, min_samples_split = 2, min_samples_leaf= 1, random_state=42)

    # Fit GridSearchCV
    rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on test set:", mse)

    to_fill = np.where(np.isnan(y_undropped))[0]
    X_with_nan = data.iloc[:, 0:16].drop('HR', axis=1)
    y_pred_with_nan = rf.predict(X_with_nan)
    plt.plot(data['HR'], label='Actual HR', color='blue', linestyle='-')
    plt.plot(y_pred_with_nan, label='Predicted HR (Linear regression)', color='red', linestyle='--')

    # Fill NaN values in 'HR' with predictions
    y_undropped[to_fill] = y_pred_with_nan[to_fill]
    data['HR'] = y_undropped
    print('yay')
    return data
    
def normalize_length_for_heatmap(data_dict, compnames):    
    data_df = pd.DataFrame(data_dict['data'])
    data_df.columns = compnames
    x = data_df.shape[0]
    n = 101
    amount_rows = ((x // n) * n + n) - x
    nan_rows = pd.DataFrame(np.nan, index = range(amount_rows), columns = data_df.columns)
    data_df = pd.concat([data_df, nan_rows], axis = 0)
    data = np.array(data_df)
    data_dict['data'] = data
    return data_dict



def process_data(data_path):
    '''Function to process the data and get it ready for the SOM'''
    # Read the data
    data_set = read_data(data_path, startcol=0, endcol=16)
    headers = read_headers(data_path)

    header_start = data_set.shape[1]
    label_names = np.array(headers[header_start:])
    compnames = np.array(headers[0:header_start])

    dataframe_labels = pd.read_csv(data_path, delimiter=',', usecols=[16])
    dataframe_labels.columns = ['Participant']
    data = data_set[:, 0:header_start]

    ID = dataframe_labels.iloc[:, 0].unique()
    start_test, end_test = get_start_end_indices(dataframe_labels)

    breath_time = []
    sData_data_df = pd.DataFrame(data)
    sData_data_df.columns = compnames

    for i in range(len(ID)):
        start = start_test[i]
        end = end_test[i]
        bt = np.diff(sData_data_df.iloc[start:end, 0].to_numpy(), prepend=0)
        breath_time.append(bt)

    sData_data_df.iloc[:, 0] = pd.Series(np.concatenate(breath_time))
    exercise_df = pd.concat([sData_data_df, dataframe_labels], axis=1)
    
    return exercise_df
    
def determine_longest_trial(data_exercise_df):
    tails = data_exercise_df.groupby('Participant').tail(1).index.tolist()
    heads = data_exercise_df.groupby('Participant').head(1).index.tolist()
    lengths = []
    for i in range(len(tails)):
        lengths.append(tails[i] - heads[i])
    longest_trial = np.max(lengths)
    return longest_trial

def Create_som(exercise_df, length_prime):
    fmv = 0
    equa = 1

    data_exercise_df = exercise_df.loc[:, exercise_df.columns != 'Participant']
    compnames = data_exercise_df.columns
    labels_exercise_df = exercise_df['Participant']
    sData_exercise = som_data_struct(data_exercise_df.to_numpy())
    sData_exercise['labels'] = labels_exercise_df.to_numpy()
    sData_exercise['label_names'] = ['Participant']
    sData_exercise['comp_names'] = compnames
    if fmv == 1:
        sData_exercise['data'] = fill_missing_values(sData_exercise['data'], compnames)

    if equa == 1:
        Equal_lengths(sData_exercise, length_prime)

    sData_exercise_copy = copy.deepcopy(sData_exercise)
    sData_exercise_norm = som_normalize(sData_exercise_copy, 'var')
    sData_exercise_norm['comp_names'] = sData_exercise_norm['comp_names'].tolist()
#    a = []
#    for x in sData_exercise_norm['labels']:
#        a.append('')
    sData_exercise_norm['labels'] = labels_exercise_df.to_numpy()
    return sData_exercise, sData_exercise_copy, sData_exercise_norm, compnames