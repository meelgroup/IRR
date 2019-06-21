import warnings
from sklearn.model_selection import train_test_split
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

warnings.simplefilter(action='ignore', category=FutureWarning)
# import h5py
# warnings.resetwarnings()
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import pickle

verbose = False


def discretization_from_file(file, colSep, rowHeader, colNames, col_y, return_column_info, fracPresent=0.9, colCateg=[],
                             numThresh=9,
                             valEq_y=None, isMultiClassData=False):
    # Quantile probabilities
    quantProb = np.linspace(1. / (numThresh + 1.), numThresh / (numThresh + 1.), numThresh)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]
    if (verbose):
        print("before discrertization: ")
        print(colNames)
        print(colCateg)
    data = pd.read_csv(file, sep=colSep, names=colNames, header=rowHeader, error_bad_lines=False)
    if (verbose):
        print(data.columns)
    if col_y == None:
        if colNames is None:
            colNames = data.columns
        col_y = colNames[-1]
        colNames = colNames[:-1]

    data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
    data.dropna(axis=0, how='any', inplace=True)

    y = data.pop(col_y).copy()
    # y = pd.DataFrame({'y': y})
    # print(y)
    if (verbose):
        print(data.shape, y.shape)
        print("\nafter discretization: ")
    # check only if multiclass data
    # Binarize if value for equality test provided
    if (not isMultiClassData):
        if valEq_y:
            y = (y == valEq_y).astype(int)

        # Ensure y is binary and contains no missing values
        assert y.nunique() == 2, "Target 'y' must be binary"
        assert y.count() == len(y), "Target 'y' must not contain missing values"
        # Rename values to 0, 1
        y.replace(np.sort(y.unique()), [0, 1], inplace=True)
    # %% Binarize features
    # Initialize dataframe and thresholds
    A = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}
    column_counter = 1
    column_set_list = []
    # Iterate over columns
    for c in data:
        # number of unique values
        # print(c)
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[(c, '', '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
            A[(c, 'not', '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

            demo = [1, column_counter, column_counter + 1]
            column_set_list.append(demo)
            column_counter += 2

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            # if (verbose):
            #     print(c)
            #     print(c in colCateg)
            #     print(data[c].dtype)
            # Dummy-code values
            Anew = pd.get_dummies(data[c]).astype(int)
            Anew.columns = Anew.columns.astype(str)
            # Append negations
            Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(c, '=='), (c, '!=')])
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

            demo = [2, column_counter, column_counter + 1]
            column_set_list.append(demo)
            column_counter += 2

        # Ordinal column
        elif np.issubdtype(data[c].dtype, int) | np.issubdtype(data[c].dtype, float):
            # Few unique values
            # if(verbose):
            #
            #     print(data[c].dtype)
            if valUniq <= numThresh + 1:
                # Thresholds are sorted unique values excluding maximum
                thresh[c] = np.sort(data[c].unique())[:-1]
            # Many unique values
            else:
                # Thresholds are quantiles excluding repetitions
                thresh[c] = data[c].quantile(q=quantProb).unique()
            # Threshold values to produce binary arrays
            Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            Anew = np.concatenate((Anew, 1 - Anew), axis=1)
            # Convert to dataframe with column labels
            Anew = pd.DataFrame(Anew,
                                columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
            # Concatenate
            # print(A.shape)
            # print(Anew.shape)
            A = pd.concat([A, Anew], axis=1)

            addedColumn = len(Anew.columns)
            addedColumn = int(addedColumn / 2)
            demo = [3]
            demo = demo + [column_counter + nc for nc in range(addedColumn)]
            column_counter += addedColumn
            column_set_list.append(demo)
            demo = [4]
            demo = demo + [column_counter + nc for nc in range(addedColumn)]
            column_counter += addedColumn
            column_set_list.append(demo)
        else:
            print(("Skipping column '" + c + "': data type cannot be handled"))
            continue

    if (return_column_info):
        return A, y, column_set_list
    return A, y


def discretization_from_direct_data(data, return_column_info, colNames=[], fracPresent=0.9, colCateg=[], numThresh=9,
                                    valEq_y=None, isMultiClassData=False):
    # Quantile probabilities
    quantProb = np.linspace(1. / (numThresh + 1.), numThresh / (numThresh + 1.), numThresh)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    data = pd.DataFrame(data.reshape(-1, len(data[0])), columns=colNames)
    data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
    data.dropna(axis=0, how='any', inplace=True)

    # check only if multiclass data
    # Binarize if value for equality test provided

    A = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}
    column_counter = 1
    column_set_list = []
    for c in data:
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[(c, '', '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
            A[(c, 'not', '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])
            demo = [1, column_counter, column_counter + 1]
            column_set_list.append(demo)
            column_counter += 2

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            Anew_train = pd.get_dummies(data[c]).astype(int)
            Anew_train.columns = Anew_train.columns.astype(str)
            Anew_train = pd.concat([Anew_train, 1 - Anew_train], axis=1, keys=[(c, '=='), (c, '!=')])
            A = pd.concat([A, Anew_train], axis=1)
            demo = [2, column_counter, column_counter + 1]
            column_set_list.append(demo)
            column_counter += 2
        # Ordinal column
        elif np.issubdtype(data[c].dtype, int) | np.issubdtype(data[c].dtype, float):
            # Few unique values
            if valUniq <= numThresh + 1:
                # Thresholds are sorted unique values excluding maximum
                thresh[c] = np.sort(data[c].unique())[:-1]
            # Many unique values
            else:
                # Thresholds are quantiles excluding repetitions
                thresh[c] = data[c].quantile(q=quantProb).unique()
            # Threshold values to produce binary arrays
            Anew_train = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            Anew_train = np.concatenate((Anew_train, 1 - Anew_train), axis=1)
            Anew_train = pd.DataFrame(Anew_train,
                                      columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
            # Concatenate
            A = pd.concat([A, Anew_train], axis=1)

            addedColumn = len(Anew_train.columns)
            addedColumn = int(addedColumn / 2)
            demo = [3]
            demo = demo + [column_counter + nc for nc in range(addedColumn)]
            column_counter += addedColumn
            column_set_list.append(demo)
            demo = [4]
            demo = demo + [column_counter + nc for nc in range(addedColumn)]
            column_counter += addedColumn
            column_set_list.append(demo)
        else:
            print(("Skipping column '" + c + "': data type cannot be handled"))
            continue

    if (return_column_info):
        return A, column_set_list
    return A


def binarizing_y_label(y):
    '''
    This function converts decimal class to binary multicolumn class
    --kind of one hot encoding for categorical features
    :param y:
    :return:
    '''
    max_y = int(math.floor(math.log(y.nlargest(1).as_matrix()[0], 2)) + 1)
    y_matrix = y.as_matrix()

    new_columns = ['y' + str(i) for i in range(max_y)]
    new_y = pd.DataFrame(columns=new_columns)
    for i in range(len(y_matrix)):
        bin_formated_str = "{0:0" + str(max_y) + "b}"
        binarized_y = bin_formated_str.format(y_matrix[i])
        new_y.loc[i] = [binarized_y[j] for j in range(max_y)]
    return new_y


def to_pickle(fname_datadump, A, A_df, y, column_set_list, col_to_feat):
    ''' save A, y, col_to_feat with auxiliary info as column_set_list
        -column_set_list is used for facilitating adjust_rule function in incremental approach
    '''

    data_dict = {'A': A, 'A_df': A_df, 'y': y, 'column_set_list': column_set_list,
                 'col_to_feat': col_to_feat}  ## tolist()
    pickle.dump(data_dict, open(fname_datadump, "wb"), protocol=2)


# def to_pickle(fname_datadump, A, A_df, y, col_to_feat):
#     ''' save A, y, col_to_feat '''
#
#     data_dict = {'A': A, 'A_df': A_df, 'y': y, 'column_set_list': column_set_list,
#                  'col_to_feat': col_to_feat}  ## tolist()
#     pickle.dump(data_dict, open(fname_datadump, "wb"), protocol=2)


def get_col_to_features_map(A_df):
    ''' produce a column to feature map '''

    cols_A = A_df.columns
    feats = [col[0] for col in cols_A]
    feat_ind = {};
    ind = 1
    for f in feats:
        if not f in feat_ind:
            feat_ind[f] = ind
            ind += 1
    signs = [col[1] for col in cols_A]
    col_to_feat = []
    for ind in range(len(feats)):
        # print "%s %s" % (feats[ind], signs[ind])
        feat_sign = "-" if signs[ind] == '>' or signs[ind] == '!=' else ''
        feat_str = "%s%d" % (feat_sign, feat_ind[feats[ind]])
        # print feat_str
        col_to_feat.append(int(feat_str))
    return col_to_feat


def save_in_disk(A, y, location, filename, column_set_list):
    col_to_feat = get_col_to_features_map(A)
    to_pickle(location + filename, A.as_matrix(), A,
              y, column_set_list, col_to_feat)
    if (verbose):
        print(location + filename)


# def save_in_disk(A, y, location, filename):
#     '''
#     Save a dataset A and corresponding class label into location (remember to add '/' at the end of the location)
#     :param A:
#     :param y:
#     :param location:
#     :param location:
#     :return:
#     '''
#     col_to_feat = get_col_to_features_map(A)
#     to_pickle(location + filename, A.as_matrix(), A,
#               y, col_to_feat)


# def partition_with_eq_prob(A, y, partition_count, location, file_name_header):
#     '''
#     Steps:
#         1. seperate data based on class value
#         2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
#         3. merge one seperate batche from each class and save
#     :param A:
#     :param y:
#     :param partition_count:
#     :param location:
#     :param file_name_header:
#     :return:
#     '''
#     max_y = y.nlargest(1).as_matrix()[0]
#     min_y = y.nsmallest(1).as_matrix()[0]
#     A_list = [[] for i in range(max_y - min_y + 1)]
#     y_list = [[] for i in range(max_y - min_y + 1)]
#     level = int(math.log(partition_count, 2.0))
#     for i in range(len(y)):
#         inserting_index = y[i]
#         y_list[inserting_index - min_y].append(y[i])
#         A_list[inserting_index - min_y].append(A.values[i].tolist())
#
#     final_partition_A_train = [[] for i in range(partition_count)]
#     final_partition_y_train = [[] for i in range(partition_count)]
#
#     for each_class in range(len(A_list)):
#         partition_list_A_train = [A_list[each_class]]
#         partition_list_y_train = [y_list[each_class]]
#
#         for i in range(level):
#             for j in range(int(math.pow(2, i))):
#                 A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
#                     partition_list_A_train[int(math.pow(2, i)) + j - 1],
#                     partition_list_y_train[int(math.pow(2, i)) + j - 1],
#                     test_size=0.5,
#                     random_state=100)
#                 partition_list_A_train.append(A_train_1)
#                 partition_list_A_train.append(A_train_2)
#                 partition_list_y_train.append(y_train_1)
#                 partition_list_y_train.append(y_train_2)
#
#         partition_list_y_train = partition_list_y_train[partition_count - 1:]
#         partition_list_A_train = partition_list_A_train[partition_count - 1:]
#
#         for i in range(partition_count):
#             final_partition_y_train[i] = final_partition_y_train[i] + partition_list_y_train[i]
#             final_partition_A_train[i] = final_partition_A_train[i] + partition_list_A_train[i]
#
#     for i in range(partition_count):
#         partition_list_y_train = pd.DataFrame({'y': final_partition_y_train[i]}).squeeze()
#         partition_list_A_train = pd.DataFrame(final_partition_A_train[i]).squeeze()
#         partition_list_A_train.columns = A.columns
#         col_to_feat = get_col_to_features_map(partition_list_A_train)
#         to_pickle(location + file_name_header + "_partition_" + str(
#             i) + ".data",
#                   partition_list_A_train.as_matrix(), partition_list_A_train,
#                   partition_list_y_train,
#                   col_to_feat)


def partition_with_eq_prob(A, y, partition_count):
    '''
        Steps:
            1. seperate data based on class value
            2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
            3. merge one seperate batche from each class and save
        :param A:
        :param y:
        :param partition_count:
        :param location:
        :param file_name_header:
        :param column_set_list: uses for incremental approach
        :return:
        '''
    # print(y)
    # max_y = y.nlargest(1).as_matrix()[0]
    # min_y = y.nsmallest(1).as_matrix()[0]
    y = y.values.ravel()
    max_y = int(y.max())
    min_y = int(y.min())
    # print(max_y,min_y)
    # print(y)
    A_list = [[] for i in range(max_y - min_y + 1)]
    y_list = [[] for i in range(max_y - min_y + 1)]
    level = int(math.log(partition_count, 2.0))
    for i in range(len(y)):
        inserting_index = int(y[i])
        y_list[inserting_index - min_y].append(y[i])
        A_list[inserting_index - min_y].append(A.values[i].tolist())

    final_partition_A_train = [[] for i in range(partition_count)]
    final_partition_y_train = [[] for i in range(partition_count)]
    for each_class in range(len(A_list)):
        partition_list_A_train = [A_list[each_class]]
        partition_list_y_train = [y_list[each_class]]

        for i in range(level):
            for j in range(int(math.pow(2, i))):
                A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                    partition_list_A_train[int(math.pow(2, i)) + j - 1],
                    partition_list_y_train[int(math.pow(2, i)) + j - 1],
                    test_size=0.5,
                    random_state=42)  # random state for keeping consistency between lp and maxsat approach
                partition_list_A_train.append(A_train_1)
                partition_list_A_train.append(A_train_2)
                partition_list_y_train.append(y_train_1)
                partition_list_y_train.append(y_train_2)

        partition_list_y_train = partition_list_y_train[partition_count - 1:]
        partition_list_A_train = partition_list_A_train[partition_count - 1:]

        for i in range(partition_count):
            final_partition_y_train[i] = final_partition_y_train[i] + partition_list_y_train[i]
            final_partition_A_train[i] = final_partition_A_train[i] + partition_list_A_train[i]

    result_A_df = []
    result_y = []
    for i in range(partition_count):
        partition_list_y_train = pd.DataFrame({'y': final_partition_y_train[i]}).squeeze()
        partition_list_A_train = pd.DataFrame(final_partition_A_train[i]).squeeze()
        partition_list_A_train.columns = A.columns
        result_A_df.append(partition_list_A_train)
        result_y.append(partition_list_y_train)
        # col_to_feat = get_col_to_features_map(partition_list_A_train)
        # to_pickle(location + file_name_header + "_partition_" + str(
        #     i) + ".data",
        #           partition_list_A_train.as_matrix(), partition_list_A_train,
        #           partition_list_y_train, column_set_list,
        #           col_to_feat)
    return result_A_df, result_y


def partition_with_eq_prob_and_store(A, y, partition_count, location, file_name_header, column_set_list):
    '''
        Steps:
            1. seperate data based on class value
            2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
            3. merge one seperate batche from each class and save
        :param A:
        :param y:
        :param partition_count:
        :param location:
        :param file_name_header:
        :param column_set_list: uses for incremental approach
        :return:
        '''
    # print(y)
    # max_y = y.nlargest(1).as_matrix()[0]
    # min_y = y.nsmallest(1).as_matrix()[0]
    y = y.values.ravel()
    max_y = int(y.max())
    min_y = int(y.min())
    # print(max_y,min_y)
    # print(y)
    A_list = [[] for i in range(max_y - min_y + 1)]
    y_list = [[] for i in range(max_y - min_y + 1)]
    level = int(math.log(partition_count, 2.0))
    for i in range(len(y)):
        inserting_index = int(y[i])
        y_list[inserting_index - min_y].append(y[i])
        A_list[inserting_index - min_y].append(A.values[i].tolist())

    final_partition_A_train = [[] for i in range(partition_count)]
    final_partition_y_train = [[] for i in range(partition_count)]
    for each_class in range(len(A_list)):
        partition_list_A_train = [A_list[each_class]]
        partition_list_y_train = [y_list[each_class]]

        for i in range(level):
            for j in range(int(math.pow(2, i))):
                A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                    partition_list_A_train[int(math.pow(2, i)) + j - 1],
                    partition_list_y_train[int(math.pow(2, i)) + j - 1],
                    test_size=0.5)
                partition_list_A_train.append(A_train_1)
                partition_list_A_train.append(A_train_2)
                partition_list_y_train.append(y_train_1)
                partition_list_y_train.append(y_train_2)

        partition_list_y_train = partition_list_y_train[partition_count - 1:]
        partition_list_A_train = partition_list_A_train[partition_count - 1:]

        for i in range(partition_count):
            final_partition_y_train[i] = final_partition_y_train[i] + partition_list_y_train[i]
            final_partition_A_train[i] = final_partition_A_train[i] + partition_list_A_train[i]

    for i in range(partition_count):
        partition_list_y_train = pd.DataFrame({'y': final_partition_y_train[i]}).squeeze()
        partition_list_A_train = pd.DataFrame(final_partition_A_train[i]).squeeze()
        partition_list_A_train.columns = A.columns
        col_to_feat = get_col_to_features_map(partition_list_A_train)
        to_pickle(location + file_name_header + "_partition_" + str(
            i) + ".data",
                  partition_list_A_train.as_matrix(), partition_list_A_train,
                  partition_list_y_train, column_set_list,
                  col_to_feat)


def partition_binary_class_data(A, y, partition_count, location, file_name_header, column_set_list):
    partition_list_A_train = [A]
    partition_list_y_train = [y]
    level = int(math.log(partition_count, 2.0))

    for i in range(level):
        for j in range(int(math.pow(2, i))):
            A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                partition_list_A_train[int(math.pow(2, i)) + j - 1],
                partition_list_y_train[int(math.pow(2, i)) + j - 1],
                test_size=0.5)
            partition_list_A_train.append(A_train_1)
            partition_list_A_train.append(A_train_2)
            partition_list_y_train.append(y_train_1)
            partition_list_y_train.append(y_train_2)

    # print(partition_count)
    # print(len(partition_list_y_train))
    partition_list_y_train = partition_list_y_train[partition_count - 1:]
    partition_list_A_train = partition_list_A_train[partition_count - 1:]
    # print(len(partition_list_A_train))

    for i in range(partition_count):
        # print(partition_list_A_train[i].shape)
        col_to_feat = get_col_to_features_map(partition_list_A_train[i])
        to_pickle(location + file_name_header + "_partition_" + str(i) + ".data",
                  partition_list_A_train[i].as_matrix(), partition_list_A_train[i],
                  partition_list_y_train[i], column_set_list,
                  col_to_feat)


# def partition_binary_class_data(A, y, partition_count, location, file_name_header):
#     partition_list_A_train = [A]
#     partition_list_y_train = [y]
#     level = int(math.log(partition_count, 2.0))
#
#     for i in range(level):
#         for j in range(int(math.pow(2, i))):
#             A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
#                 partition_list_A_train[int(math.pow(2, i)) + j - 1],
#                 partition_list_y_train[int(math.pow(2, i)) + j - 1],
#                 test_size=0.5,
#                 random_state=100)
#             partition_list_A_train.append(A_train_1)
#             partition_list_A_train.append(A_train_2)
#             partition_list_y_train.append(y_train_1)
#             partition_list_y_train.append(y_train_2)
#
#     print(partition_count)
#     # print(len(partition_list_y_train))
#     partition_list_y_train = partition_list_y_train[partition_count - 1:]
#     partition_list_A_train = partition_list_A_train[partition_count - 1:]
#     # print(len(partition_list_A_train))
#
#     for i in range(partition_count):
#         print(partition_list_A_train[i].shape)
#         col_to_feat = get_col_to_features_map(partition_list_A_train[i])
#         to_pickle(location + file_name_header + "_partition_" + str(i) + ".data",
#                   partition_list_A_train[i].as_matrix(), partition_list_A_train[i],
#                   partition_list_y_train[i],
#                   col_to_feat)


def do_k_fold(X, y, k):
    y = y.to_frame().reset_index(drop=True)

    kf = StratifiedKFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X, y)
    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    for train_index, test_index in kf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)

        X_trains.append(X.loc[train_index, :])
        X_tests.append(X.loc[test_index, :])
        y_trains.append(y.loc[train_index, :])
        y_tests.append(y.loc[test_index, :])

        # print(X_train.shape, X_test.shape)
        # print(y_train.shape, y_test.shape)
        # save_in_disk(X_train, y_train, save_location, dataset + "_train_" + str(cnt) + ".data", column_info)
        # save_in_disk(X_test, y_test, save_location, dataset + "_test_" + str(cnt) + ".data", column_info)

    return X_trains, y_trains, X_tests, y_tests
