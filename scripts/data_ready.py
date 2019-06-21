import re
import csv
# import iterative_learning
# import iterative_learning_with_counterfactual
import util
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectKBest
# from other_classifier import call_classifier
import os
import time
import argparse
# import imli_new_obj
import pickle


def toArff(X, y, columns, dataset, save_location="../Benchmarks/tempfiles/", runIndex=0, test_or_train="train"):
    X = X.astype(int)
    y = y.astype(int)
    file = open(save_location + str(dataset) + "_" +
                str(test_or_train)+"_" + str(runIndex)+".arff", "w")
    file.write("@relation " + str(dataset))
    file.write("\n\n")
    for column in columns:
        file.write("@attribute " +
                   str(column.replace(" ", "_")) + str(" {0,1}\n"))
    file.write("@attribute " + "target" + str(" {0,1}\n"))
    file.write("\n\n@data\n")
    for cnt in range(len(X)):
        for elem in X[cnt]:
            file.write(str(int(elem))+",")
        file.write(str(int(y[cnt]))+"\n")
    file.close()
    return


def to_tabular_form_brl(Xtrain, ytrain, Xtest, ytest, columns, dataset, save_location="../benchmarks/tempfiles/", runIndex=0):

    fout = open(save_location+dataset+"_"+str(runIndex)+"_train"+".tab", "w")
    for row in Xtrain:
        cnt = 0
        row_str = ""
        for value in row:
            row_str += (columns[cnt]+"_"+str(int(value))+" ")
            cnt += 1
        fout.write(row_str[:-1]+"\n")
    fout.close

    fout = open(save_location+dataset+"_"+str(runIndex)+"_train"+".Y", "w")
    for value in ytrain:
        if(value == 1):
            fout.write("0 1\n")
        else:
            fout.write("1 0\n")

    fout.close

    fout = open(save_location+dataset+"_"+str(runIndex)+"_test"+".tab", "w")
    for row in Xtest:
        cnt = 0
        row_str = ""
        for value in row:
            row_str += (columns[cnt]+"_"+str(int(value))+" ")
            cnt += 1
        fout.write(row_str[:-1]+"\n")
    fout.close

    fout = open(save_location+dataset+"_"+str(runIndex)+"_test"+".Y", "w")
    for value in ytest:
        if(value == 1):
            fout.write("0 1\n")
        else:
            fout.write("1 0\n")

    fout.close

    return


def to_format_BOA_model(Xtrain, ytrain, Xtest, ytest, columns, dataset, save_location="../benchmarks/tempfiles/", runIndex=0):

    header = ""
    for column in columns:
        if("_eq_" in column):
            column = column.replace("_eq_", "=@")
        if("_l_" in column):
            column = column.replace("_l_", "<@")
        if("_ge_" in column):
            column = column.replace("_ge_", ">=@")
        column = column.replace("_", "").replace("@", "_")
        column = re.sub(r"\s", "", column)
        # print(column)
        header += column+" "
    # print(header)

    fout = open(save_location+dataset+"_" +
                str(runIndex)+"_Xtrain_BOA"+".txt", "w")
    fout.write(header[:-1]+"\n")
    for row in Xtrain:
        cnt = 0
        row_str = ""
        for value in row:
            row_str += str(int(value))+" "
            cnt += 1
        fout.write(row_str[:-1]+"\n")
    fout.close

    fout = open(save_location+dataset+"_" +
                str(runIndex)+"_ytrain_BOA"+".txt", "w")
    for value in ytrain:
        fout.write(str(int(value))+"\n")

    fout.close

    fout = open(save_location+dataset+"_" +
                str(runIndex)+"_Xtest_BOA"+".txt", "w")
    fout.write(header[:-1]+"\n")

    for row in Xtest:
        cnt = 0
        row_str = ""
        for value in row:
            row_str += str(int(value))+" "
            cnt += 1
        fout.write(row_str[:-1]+"\n")
    fout.close

    fout = open(save_location+dataset+"_" +
                str(runIndex)+"_ytest_BOA"+".txt", "w")
    for value in ytest:
        fout.write(str(int(value))+"\n")

    fout.close

    return


def runExperiment():

    datasets = ["adult", "credit", "ionosphere", "iris",
                "parkinsons", "pima", "toms", "transfusion", "twitter", "wdbc", "tictactoe", "compas", "titanic", "iris", "heart", "ilpd"]

    for dataset in datasets:
        model = util.imli()

        X, y, X_cat, y_cat = model.discretize_orange(
            "../Benchmarks/Data/"+dataset+".csv")

        columns = model.getColumns()
        categorical_columns = model.categorical_columns
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        skf.get_n_splits(X, y)
        
        # print("if(key == \'"+dataset+"\'):")
        # print("\tsize.append("+str(len(X))+")")
        # print("\tfeatures.append("+str(len(columns))+")")
        # print("\tdatasets.append(\""+dataset+"\")")
        
        cnt = 0
        for train, test in skf.split(X, y):

            # pickle format for general classifiers
            data_dict = {'Xtrain': X[train],  'ytrain': y[train],
                         'Xtest': X[test], 'ytest': y[test], 'columns': columns}
            pickle.dump(data_dict, open("../Benchmarks/tempfiles/" +
                                        dataset+"_"+str(cnt)+".dat", "wb"), protocol=2)

            # arff for ripper
            toArff(X[train], y[train], columns, dataset,
                   runIndex=cnt, test_or_train="train")
            toArff(X[test], y[test], columns, dataset,
                   runIndex=cnt, test_or_train="test")

            # tabular format for bayesian rule lists
            # to_tabular_form_brl(X_cat[train], y_cat[train],
            #                     X_cat[test], y_cat[test], categorical_columns, dataset, runIndex=cnt)

            # to_format_BOA_model(X[train], y[train], X[test], y[test], columns, dataset,
            #                     runIndex=cnt)
            cnt += 1


runExperiment()
