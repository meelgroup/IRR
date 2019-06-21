# -*- coding: utf-8 -*-
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import warnings
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import Orange
warnings.simplefilter(action='ignore', category=FutureWarning)


class imli():
    def __init__(self, numPartition=-1, numClause=1, lambda_param=.5, solver="open-wbo", ruleType="CNF",
                 workDir=".", targetClass=1, subSampleSize=64, itrConstant=1):
        '''

        :param numPartition: no of partitions of training dataset
        :param numClause: no of clause in the formulaf
        :param dataFidelity: weight corresponding to accuracy
        :param weightFeature: weight corresponding to selected features
        :param solver: specify the (name of the) bin of the solver; bin must be in the path
        :param ruleType: type of rule {CNF,DNF}
        :param workDir: working directory
        :param verbose: True for debug
        '''

        self.numPartition = numPartition
        self.numClause = numClause
        self.lambda_param = lambda_param
        # self.dataFidelity = dataFidelity
        # self.weightFeature = weightFeature
        self.solver = solver
        self.ruleType = ruleType
        self.workDir = workDir
        self.verbose = False  # not necessary
        # self.trainingError = 0
        self.selectedFeatureIndex = []
        self.columns = []
        self.applyProb = False
        self.applySamling = True
        self.applyNewObjective = False
        self.dataFidelity_current = 0.5
        self.targetClass = targetClass
        self.prune_features = True
        self.subSampleSize = subSampleSize
        self.itr_constant = itrConstant

    def __repr__(self):
        return "<imli numPartition:%s numClause:%s regulrization:%s " \
               "solver:%s ruleType:%s workDir:%s>" % (self.numPartition,
                                                      self.numClause, self.lambda_param, self.solver,
                                                      self.ruleType, self.workDir)

    def getColumns(self):
        # return_columns = []
        # for column in self.columns:
        #     column = column.replace('_l_', ' < ')
        #     column = column.replace('_ge_', ' >= ')
        #     column = column.replace('_eq_', ' = ')
        #     return_columns.append(column)
        # return return_columns
        return self.columns
    def __prune_features(self, X, y, threshold=0.5, forTrain=True):

        if(forTrain):
            column_one_negative = [0 for i in self.columns]
            column_one_positive = [0 for i in self.columns]
            column_one = [0 for i in self.columns]
            for i in range(len(y)):
                for j in range(len(X[i])):
                    if(X[i][j] == 1 and self.ruleType == "CNF"):
                        if(y[i] == 0):
                            column_one_negative[j] += 1
                            column_one[j] -= 1
                        else:
                            column_one_positive[j] += 1
                            column_one[j] += 1
                    if(X[i][j] == 0 and self.ruleType == "DNF"):
                        if(y[i] == 0):
                            column_one_negative[j] += 1
                            column_one[j] -= 1
                        else:
                            column_one_positive[j] += 1
                            column_one[j] += 1

            column_rank_negative = rankdata(
                column_one_negative, method='ordinal')
            column_rank_positive = (
                len(self.columns)+1-rankdata(column_one_positive, method='ordinal')).astype(int)
            column_rank = (
                len(self.columns)+1-rankdata(column_one, method='ordinal')).astype(int)
            self.reduced_column_index = []
            reduced_columns = []
            threshold = threshold*len(self.columns)
            for i in range(len(self.columns)):
                # print(i,  column_rank[i], column_one[i])
                if(column_rank[i] <= threshold or column_one[i] > 0):
                    reduced_columns.append(self.columns[i])
                    self.reduced_column_index.append(i)
            # print(reduced_column_index)
            self.columns = reduced_columns
        newX = []
        for i in range(len(X)):
            newX.append([])
            for j in self.reduced_column_index:
                newX[i].append(X[i][j])

        return newX

    def erase(self):
        self.selectedFeatureIndex = []
        self.assignList = []
        self.dataFidelity_current = 0.5

    def getSelectedColumnIndex(self):
        return_list = [[] for i in range(self.numClause)]
        ySize = len(self.columns)
        # print(self.selectedFeatureIndex)
        for elem in self.selectedFeatureIndex:
            new_index = int(elem)-1
            return_list[int(new_index/ySize)].append(new_index % ySize)
        # print(return_list)
        return return_list

    def getNumOfPartition(self):
        return self.numPartition

    def getNumOfClause(self):
        return self.numClause

    def getWeightFeature(self):
        return self.weightFeature

    def getRuleType(self):
        return self.ruleType

    def getWorkDir(self):
        return self.workDir

    def getWeightDataFidelity(self):
        return self.dataFidelity

    def getSolver(self):
        return self.solver

    def getRuleSize(self):
        return len(self.selectedFeatureIndex)

    def discretize_orange(self, csv_file):
        data = Orange.data.Table(csv_file)
        # Run impute operation for handling missing values
        imputer = Orange.preprocess.Impute()
        data = imputer(data)
        # Discretize datasets
        discretizer = Orange.preprocess.Discretize()
        discretizer.method = Orange.preprocess.discretize.EntropyMDL(
            force=False)
        discetized_data = discretizer(data)
        self.categorical_columns=[elem.name for  elem in discetized_data.domain[:-1]]
        # Apply one hot encoding on X (Using continuizer of Orange)
        continuizer = Orange.preprocess.Continuize()
        binarized_data = continuizer(discetized_data)
        # self.columns = [binarized_data.domain[i].name for i in range(
        #     len(binarized_data.domain)-1)]
        # print(self.columns)
        # self.columns = []
        for i in range(len(binarized_data.domain)-1):
            column = binarized_data.domain[i].name
            if("<" in column):
                column = column.replace("=<", "_l_")
            elif("≥" in column):
                column = column.replace("=≥", "_ge_")
            elif("=" in column):
                if("-" in column):
                    column = column.replace("=", "_eq_(")
                    column = column+")"
                else:
                    column = column.replace("=", "_eq_")
                    column = column
            self.columns.append(column)
        # print(self.columns)
        return binarized_data.X, binarized_data.Y,discetized_data.X,discetized_data.Y

    def fit(self, XTrain, yTrain, threshold=0.5):

        # self.trainingError = 0

        if(self.prune_features):
            XTrain = self.__prune_features(XTrain, yTrain, threshold=threshold)

        self.dataFidelity_current = 0.5
        self.trainingSize = len(XTrain)

        if(not self.applySamling and self.numPartition == -1):
            self.numPartition = 2**math.floor(
                math.log2(len(XTrain)/self.subSampleSize))

        if(not self.applySamling):
            XTrains, yTrains = self.partitionWithEqualProbability(
                XTrain, yTrain)

        if(self.applySamling):
            self.importanceVector = [
                float(i/self.trainingSize) for i in range(self.trainingSize)]
            if(self.numPartition == -1):
                self.numPartition = 10

            # print(self.importanceVector)

        self.assignList = []
        z_cur = 0
        for each_partition in range(self.numPartition):
            prev_assignList = self.assignList
            z_prev = z_cur

            if(not self.applySamling):
                # set weight
                gcd = math.gcd(
                    self.numClause*len(XTrains[each_partition][0]), len(XTrains[each_partition]))
                self.weightFeature = int(
                    (100000*gcd*(1-self.lambda_param))/(self.numClause*len(XTrains[each_partition][0])))+1
                self.dataFidelity = int(
                    (100000*gcd*self.lambda_param)/(len(XTrains[each_partition])))+1

                self.learnModel(XTrains[each_partition],
                                yTrains[each_partition], isTest=False)

            if(self.applySamling):

                XTrain_sampled, yTrain_sampled = self.generatSamples(
                    XTrain, yTrain)

                # consider full training set while setting noise weight

                gcd = math.gcd(
                    self.numClause*len(XTrain_sampled[0]), self.trainingSize)

                exponent = each_partition/(self.itr_constant*self.numPartition)
                self.weightFeature = int(
                    (100000*gcd*(1-math.pow(self.lambda_param, exponent))) /
                    (self.numClause*len(XTrain_sampled[0])))+1
                self.dataFidelity = int(
                    (100000*gcd*math.pow(self.lambda_param, exponent))/self.trainingSize)+1

                # print(self.weightFeature)
                # print(self.dataFidelity)

                self.learnModel(XTrain_sampled,
                                yTrain_sampled, isTest=False)

                if(each_partition != 0):
                    self.dataFidelity_current = self.dataFidelity_current - \
                        0.5 / (self.numPartition-1)
                # print(self.dataFidelity_current)

            if(each_partition != 0 and self.applyProb == True):

                # compute z (only on accuracy)
                z_cur = accuracy_score(self.predict(XTrain, yTrain), yTrain)
                self.acc_partition = z_cur
                # calculate probability and revert if current solution is not good enough
                prob = 1.0/(1.0+math.exp(z_cur*100-z_prev*100))
                rand = int.from_bytes(os.urandom(
                    8), byteorder="big") / ((1 << 64) - 1)
                if(rand <= prob):
                    z_cur = z_prev
                    self.acc_partition = z_cur
                    self.assignList = prev_assignList

    def generatSamples(self, XTrain, yTrain):
        list_of_random_index = random.sample(
            [i for i in range(self.trainingSize)], self.subSampleSize)
        XTrain_sampled = [XTrain[i] for i in list_of_random_index]
        yTrain_sampled = [yTrain[i] for i in list_of_random_index]
        return XTrain_sampled, yTrain_sampled

    def predict(self, XTest, yTest):
        if(self.prune_features):
            XTest = self.__prune_features(XTest, yTest, forTrain=False)
        predictions = self.learnModel(XTest, yTest, isTest=True)
        yhat = []
        for i in range(len(predictions)):
            if (int(predictions[i]) > 0):
                yhat.append(1 - yTest[i])
            else:
                yhat.append(yTest[i])
        return yhat

    def learnModel(self, X, y, isTest):

        # temp files to save maxsat query in wcnf format
        WCNFFile = self.workDir + "/" + "model.wcnf"
        outputFileMaxsat = self.workDir + "/" + "model_out.txt"

        # generate maxsat query for dataset
        if (self.ruleType == 'DNF'):
            #  negate yVector for DNF rules
            self.generateWCNFFile(X, [1 - int(y[each_y]) for each_y in
                                      range(len(y))],
                                  len(X[0]), WCNFFile,
                                  isTest)

        else:
            self.generateWCNFFile(X, y, len(X[0]),
                                  WCNFFile,
                                  isTest)

        # call a maxsat solver

        cmd = self.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat
        os.system(cmd)

        # delete temp files
        cmd = "rm " + WCNFFile
        os.system(cmd)

        # parse result of maxsat solving
        f = open(outputFileMaxsat, 'r')
        lines = f.readlines()
        f.close()
        optimumFound = False
        bestSolutionFound = False
        solution = ''
        for line in lines:
            if (self.solver == "maxroster" and line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break
            elif (self.solver == "LMHS" and line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break
            elif (self.solver == "qmaxsat" and line.strip().startswith('v')):
                solution = solution + " " + line.strip().strip('v ')

            elif (line.strip().startswith('v') and optimumFound):
                solution = line.strip().strip('v ')
                break
            elif (line.strip().startswith('c ') and bestSolutionFound):
                solution = line.strip().strip('c ')
                break
            elif (line.strip().startswith('s OPTIMUM FOUND')):
                optimumFound = True
                # if (self.verbose):
                #     print("Optimum solution found")
            elif (line.strip().startswith('c Best Model Found:')):
                bestSolutionFound = True
                # if (self.verbose):
                #     print("Best solution found")

        fields = solution.split()
        TrueRules = []
        TrueErrors = []
        zeroOneSolution = []

        # fields = self.pruneRules(fields, len(X[0]))
        for field in fields:
            if (int(field) > 0):
                zeroOneSolution.append(1.0)
            else:
                zeroOneSolution.append(0.0)
            if (int(field) > 0):

                if (abs(int(field)) <= self.numClause * len(X[0])):

                    TrueRules.append(field)
                elif (self.numClause * len(X[0]) < abs(int(field)) <= self.numClause * len(
                        X[0]) + len(y)):
                    TrueErrors.append(field)

        # if (self.verbose):
        #     print("The number of True Rule are: " + str(len(TrueRules)))
        #     print("The number of errors are:    " + str(len(TrueErrors)) + " out of " + str(len(y)))
        self.xhat = []

        for i in range(self.numClause):
            self.xhat.append(np.array(
                zeroOneSolution[i * len(X[0]):(i + 1) * len(X[0])]))
        err = np.array(zeroOneSolution[len(X[0]) * self.numClause: len(
            X[0]) * self.numClause + len(y)])

        # delete temp files
        cmd = "rm " + outputFileMaxsat
        os.system(cmd)

        if (not isTest):
            self.assignList = fields[:self.numClause * len(X[0])]
            # self.trainingError += len(TrueErrors)
            self.selectedFeatureIndex = TrueRules

        return fields[self.numClause * len(X[0]):len(y) + self.numClause * len(X[0])]

    def partitionWithEqualProbability(self, X, y):
        '''
            Steps:
                1. seperate data based on class value
                2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
                3. merge one seperate batche from each class and save
            :param X:
            :param y:
            :param partition_count:
            :param location:
            :param file_name_header:
            :param column_set_list: uses for incremental approach
            :return:
            '''
        partition_count = self.numPartition
        # y = y.values.ravel()
        max_y = int(max(y))
        min_y = int(min(y))

        X_list = [[] for i in range(max_y - min_y + 1)]
        y_list = [[] for i in range(max_y - min_y + 1)]
        level = int(math.log(partition_count, 2.0))
        for i in range(len(y)):
            inserting_index = int(y[i])
            y_list[inserting_index - min_y].append(y[i])
            X_list[inserting_index - min_y].append(X[i])

        final_partition_X_train = [[] for i in range(partition_count)]
        final_partition_y_train = [[] for i in range(partition_count)]
        for each_class in range(len(X_list)):
            partition_list_X_train = [X_list[each_class]]
            partition_list_y_train = [y_list[each_class]]

            for i in range(level):
                for j in range(int(math.pow(2, i))):
                    A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                        partition_list_X_train[int(math.pow(2, i)) + j - 1],
                        partition_list_y_train[int(math.pow(2, i)) + j - 1],
                        test_size=0.5)  # random state for keeping consistency between lp and maxsat approach
                    partition_list_X_train.append(A_train_1)
                    partition_list_X_train.append(A_train_2)
                    partition_list_y_train.append(y_train_1)
                    partition_list_y_train.append(y_train_2)

            partition_list_y_train = partition_list_y_train[partition_count - 1:]
            partition_list_X_train = partition_list_X_train[partition_count - 1:]

            for i in range(partition_count):
                final_partition_y_train[i] = final_partition_y_train[i] + \
                    partition_list_y_train[i]
                final_partition_X_train[i] = final_partition_X_train[i] + \
                    partition_list_X_train[i]

        return final_partition_X_train[:partition_count], final_partition_y_train[:partition_count]

    def learnSoftClauses(self, isTestPhase, xSize, yVector):
        cnfClauses = ''
        numClauses = 0

        if (isTestPhase):

            topWeight = len(yVector) + 1
            numClauses = 0

            # for noise variables
            for i in range(self.numClause * xSize + 1, self.numClause * xSize + len(yVector) + 1):
                numClauses += 1
                cnfClauses += str(1) + ' ' + str(-i) + ' 0\n'

            # for testing, the positive assigned feature variables are converted to hard clauses
            # so that  their assignment is kept consistent and only noise variables are considered soft,
            for each_assign in self.assignList:
                numClauses += 1
                cnfClauses += str(topWeight) + ' ' + each_assign + ' 0\n'
        else:

            # prev objective function, which is |R_i|-|R_i|+ lambda|E_R|

            if(not self.applyNewObjective):

                # applicable for the 1st partition

                isEmptyAssignList = True

                total_additional_weight = 0

                for each_assign in self.assignList:
                    isEmptyAssignList = False
                    numClauses += 1
                    if (int(each_assign) > 0):

                        cnfClauses += str(self.weightFeature) + \
                            ' ' + each_assign + ' 0\n'
                        total_additional_weight += self.weightFeature

                    else:
                        cnfClauses += str(self.weightFeature) + \
                            ' ' + each_assign + ' 0\n'
                        total_additional_weight += self.weightFeature

                # noise variables are to be kept consisitent (not necessary though)
                for i in range(self.numClause * xSize + 1,
                               self.numClause * xSize + len(yVector) + 1):
                    numClauses += 1
                    cnfClauses += str(self.dataFidelity) + \
                        ' ' + str(-i) + ' 0\n'

                # for the first step
                if (isEmptyAssignList):

                    for i in range(1, self.numClause * xSize + 1):
                        numClauses += 1
                        cnfClauses += str(self.weightFeature) + \
                            ' ' + str(-i) + ' 0\n'
                        total_additional_weight += self.weightFeature

                topWeight = int(self.dataFidelity * len(yVector) +
                                1 + total_additional_weight)
            else:
                # new objective function, which is |R_i| + lambda|E_R| + lambda|E_(R-1)|

                total_additional_weight = 0
                isEmptyAssignList = True
                for each_assign in self.assignList:
                    isEmptyAssignList = False
                    numClauses += 1

                    # variables, who are in the previous rule
                    if (int(each_assign) > 0):
                        # print(each_assign)
                        # do nothing
                        cnfClauses = cnfClauses
                        total_additional_weight += 0

                        # cnfClauses += str(self.weightFeature) + \
                        #     ' ' + each_assign + ' 0\n'
                        # total_additional_weight += self.weightFeature

                    else:
                        cnfClauses += str(self.weightFeature) + \
                            ' ' + each_assign + ' 0\n'
                        total_additional_weight += self.weightFeature

                # for the first step
                if (isEmptyAssignList):
                    # print("once")
                    for i in range(1, self.numClause * xSize + 1):
                        numClauses += 1
                        cnfClauses += str(self.weightFeature) + \
                            ' ' + str(-i) + ' 0\n'
                        total_additional_weight += self.weightFeature

                # UnEqual penalty for classes

                # print(len(yVector))
                cardTargetClass = 0
                for i in range(len(yVector)):
                    if yVector[i] == self.targetClass:
                        cardTargetClass += 1
                cardNegTargetClass = len(yVector)-cardTargetClass

                # noise variables are to be kept consisitent (not necessary though)
                cnt = 0
                noiseWeight = 0
                for i in range(self.numClause * xSize + 1,
                               self.numClause * xSize + len(yVector) + 1):
                    numClauses += 1
                    if(yVector[cnt] == self.targetClass):

                        if(len(self.selectedFeatureIndex) != 0):
                            wt = int(
                                (self.dataFidelity*self.dataFidelity_current*(len(yVector))/(2*cardTargetClass)))+1

                            # print(yVector[cnt], wt)
                            noiseWeight += wt
                            cnfClauses += str(wt) + \
                                ' ' + str(-i) + ' 0\n'
                        else:

                            wt = int(
                                (self.dataFidelity*(len(yVector))/(2*cardTargetClass)))
                            noiseWeight += wt
                            # print(yVector[cnt], wt)

                            cnfClauses += str(wt) + \
                                ' ' + str(-i) + ' 0\n'
                    else:
                        if(len(self.selectedFeatureIndex) != 0):
                            wt = int((self.dataFidelity*self.dataFidelity_current *
                                      (len(yVector))/(2*cardNegTargetClass)))+1
                            noiseWeight += wt
                            # print(yVector[cnt], wt)

                            cnfClauses += str(wt) + \
                                ' ' + str(-i) + ' 0\n'
                        else:
                            wt = int(
                                (self.dataFidelity*(len(yVector))/(2*cardNegTargetClass)))
                            noiseWeight += wt
                            # print(yVector[cnt], wt)

                            cnfClauses += str(wt) + \
                                ' ' + str(-i) + ' 0\n'

                    cnt += 1

                if(len(self.selectedFeatureIndex) != 0):
                    topWeight = noiseWeight + 1 + total_additional_weight
                else:
                    topWeight = noiseWeight + 1 + total_additional_weight

                previous_rule = self.getSelectedColumnIndex()

                # encode the previous rule
                for i in range(self.numClause):
                    if(len(previous_rule[i]) > 0):
                        clauseWeight = int((
                            self.dataFidelity*self.acc_partition*len(yVector)*(1-self.dataFidelity_current)+1))
                        newClause = str(clauseWeight)
                        # print(self.acc_partition)
                        # print(self.dataFidelity)
                        # print(len(yVector))
                        for j in range(len(previous_rule[i])):
                            newClause += " " + \
                                str(i*xSize+previous_rule[i][j]+1)
                        newClause += " 0\n"
                        cnfClauses += newClause
                        topWeight += clauseWeight

        return topWeight, numClauses, cnfClauses

    def generateWCNFFile(self, AMatrix, yVector, xSize, WCNFFile,
                         isTestPhase):
        # learn soft clauses associated with feature variables and noise variables
        topWeight, numClauses, cnfClauses = self.learnSoftClauses(isTestPhase, xSize,
                                                                  yVector)

        # learn hard clauses,
        additionalVariable = 0
        for i in range(len(yVector)):
            noise = self.numClause * xSize + i + 1

            # implementation of tseitin encoding
            if (yVector[i] != self.targetClass):
                new_clause = str(topWeight) + " " + str(noise)
                for each_level in range(self.numClause):
                    new_clause += " " + \
                        str(additionalVariable + each_level +
                            len(yVector) + self.numClause * xSize + 1)
                new_clause += " 0\n"
                cnfClauses += new_clause
                numClauses += 1

                for each_level in range(self.numClause):
                    for j in range(len(AMatrix[i])):
                        if (int(AMatrix[i][j]) == 1):
                            numClauses += 1
                            new_clause = str(topWeight) + " -" + str(
                                additionalVariable + each_level + len(yVector) + self.numClause * xSize + 1)

                            new_clause += " -" + \
                                str(int(j + each_level * xSize + 1))
                            new_clause += " 0\n"
                            cnfClauses += new_clause
                additionalVariable += self.numClause

            else:
                for each_level in range(self.numClause):
                    numClauses += 1
                    new_clause = str(topWeight) + " " + str(noise)
                    for j in range(len(AMatrix[i])):
                        if (int(AMatrix[i][j]) == 1):
                            new_clause += " " + \
                                str(int(j + each_level * xSize + 1))
                    new_clause += " 0\n"
                    cnfClauses += new_clause
        # print(cnfClauses)


        # write in wcnf format
        header = 'p wcnf ' + str(additionalVariable + xSize * self.numClause + (len(yVector))) + ' ' + str(
            numClauses) + ' ' + str(topWeight) + '\n'
        f = open(WCNFFile, 'w')
        f.write(header)
        f.write(cnfClauses)
        f.close()

    def getRule(self):
        generatedRule = '( '

        for i in range(self.numClause):
            inds_nnz = self.getSelectedColumnIndex()[i]

            str_clauses = [''.join(self.columns[ind]) for ind in inds_nnz]
            if (self.ruleType == "CNF"):
                rule_sep = ' %s ' % "OR"
            else:
                rule_sep = ' %s ' % "AND"
            rule_str = rule_sep.join(str_clauses)
            if (self.ruleType == 'DNF'):
                rule_str = rule_str.replace('_l_', ' >= ')
                rule_str = rule_str.replace('_ge_', ' < ')
                rule_str = rule_str.replace('_eq_', ' != ')
                # rule_str = rule_str.replace('is', '??').replace(
                #     'is not', 'is').replace('??', 'is not')
            if (self.ruleType == 'CNF'):
                rule_str = rule_str.replace('_l_', ' < ')
                rule_str = rule_str.replace('_ge_', ' >= ')
                rule_str = rule_str.replace('_eq_', ' = ')

            generatedRule += rule_str
            if (i < self.numClause - 1):
                if (self.ruleType == "DNF"):
                    generatedRule += ' ) OR \n( '
                if (self.ruleType == 'CNF'):
                    generatedRule += ' ) AND \n( '
        generatedRule += ' )'

        return generatedRule
