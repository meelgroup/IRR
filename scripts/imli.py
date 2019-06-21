import time
import sys
import os
import argparse
import pickle
import numpy
import multiprocessing
import itertools
from subprocess import STDOUT, check_output
from sat import parseFiles, generate_WCNF_file_with_card_constraints_for_all__samples, \
    generate_WCNF_file_with_card_constraint_per_sample, \
    generate_WCNF_file_with_card_constraint_per_sample
# from helper import partition_with_eq_prob
import math
from sklearn.model_selection import train_test_split


verbose = not True


def learnSoftClauses(isTestPhase, weightRegularization, weightFeature, xSize, level, yVector, assignList,
                     previousSampleCount):
    cnfClauses = ''
    numClauses = 0

    if (isTestPhase):
        topWeight = weightRegularization * \
            len(yVector) + 1 + weightFeature * xSize * level
        numClauses = 0
        for i in range(1, level * xSize + 1):
            numClauses += 1
            cnfClauses += str(weightFeature) + ' ' + str(-i) + ' 0\n'
        for i in range(level * xSize + 1, level * xSize + len(yVector) + 1):
            numClauses += 1
            cnfClauses += str(weightRegularization) + ' ' + str(-i) + ' 0\n'

        # for testing, the positive assigned feature variables are converted to hard clauses
        # so that  their assignment is kept consistent and only noise variables are considered soft,
        for each_assign in assignList:
            numClauses += 1
            cnfClauses += str(topWeight) + ' ' + each_assign + ' 0\n'
    else:
        # applicable for the 1st partition
        isEmptyAssignList = True

        total_additional_weight = 0
        positiveLiteralWeight = weightFeature
        for each_assign in assignList:
            isEmptyAssignList = False
            numClauses += 1
            if (int(each_assign) > 0):

                cnfClauses += str(positiveLiteralWeight) + \
                    ' ' + each_assign + ' 0\n'
                total_additional_weight += positiveLiteralWeight

            else:
                cnfClauses += str(weightFeature) + ' ' + each_assign + ' 0\n'
                total_additional_weight += weightFeature

        # noise variables are to be kept consisitent (not necessary though)
        for i in range(level * xSize + 1 + previousSampleCount,
                       level * xSize + len(yVector) + 1 + previousSampleCount):
            numClauses += 1
            cnfClauses += str(weightRegularization) + ' ' + str(-i) + ' 0\n'

        # for the first step
        if (isEmptyAssignList):
            for i in range(1, level * xSize + 1):
                numClauses += 1
                cnfClauses += str(weightFeature) + ' ' + str(-i) + ' 0\n'
                total_additional_weight += weightFeature

        topWeight = int(weightRegularization * len(yVector) +
                        1 + total_additional_weight)

    return topWeight, numClauses, cnfClauses


def generateWCNFFile(AMatrix, yVector, weightRegularization, weightFeature, xSize, level, WCNFFile, assignList,
                     previousSampleCount, isTestPhase):
    # learn soft clauses associated with feature variables and noise variables
    topWeight, numClauses, cnfClauses = learnSoftClauses(isTestPhase, weightRegularization, weightFeature, xSize, level,
                                                         yVector,
                                                         assignList, previousSampleCount)

    # learn hard clauses,
    additionalVariable = 0
    for i in range(len(yVector)):
        noise = level * xSize + i + 1 + previousSampleCount

        # implementation of tseitin encoding
        if (yVector[i] == 0):
            new_clause = str(topWeight) + " " + str(noise)
            for each_level in range(level):
                new_clause += " " + \
                    str(additionalVariable + each_level +
                        len(yVector) + level * xSize + 1)
            new_clause += " 0\n"
            cnfClauses += new_clause
            numClauses += 1

            for each_level in range(level):
                for j in range(len(AMatrix[i])):
                    if (int(AMatrix[i][j]) == 1):
                        numClauses += 1
                        new_clause = str(topWeight) + " -" + str(
                            additionalVariable + each_level + len(yVector) + level * xSize + 1)

                        new_clause += " -" + \
                            str(int(j + each_level * xSize + 1))
                        new_clause += " 0\n"
                        cnfClauses += new_clause
            additionalVariable += level

        else:
            for each_level in range(level):
                numClauses += 1
                new_clause = str(topWeight) + " " + str(noise)
                for j in range(len(AMatrix[i])):
                    if (int(AMatrix[i][j]) == 1):
                        new_clause += " " + \
                            str(int(j + each_level * xSize + 1))
                new_clause += " 0\n"
                cnfClauses += new_clause

    # write in wcnf format
    header = 'p wcnf ' + str(additionalVariable + previousSampleCount + xSize * level + (len(yVector))) + ' ' + str(
        numClauses) + ' ' + str(topWeight) + '\n'
    f = open(WCNFFile, 'w')
    f.write(header)
    f.write(cnfClauses)
    f.close()


def recoverRule(columns, x_hat_vec, ruleType, level, m):
    # print(columns)

    compoundStr = '( '
    for i in range(level):
        x_hat = x_hat_vec[i]
        inds_nnz = numpy.where(abs(x_hat) > 1e-4)[0]
        str_clauses = [''.join(columns[ind]) for ind in inds_nnz]
        # rule_sep = ' %s ' % ruleType
        rule_sep = ' + '

        rule_str = rule_sep.join(str_clauses)
        if (ruleType == 'and'):
            rule_str = rule_str.replace('<=', '??').replace(
                '>', '<=').replace('??', '>')
        compoundStr += rule_str
        compoundStr += ' )>=' + str(m)

        if (i < level - 1):
            if (ruleType == 'and'):
                compoundStr += ' or\n( '
            if (ruleType == 'or'):
                compoundStr += ' and\n( '
    
    # print(compoundStr)
    compoundStr = compoundStr.replace(" ", "^^")
    compoundStr = compoundStr.replace("(", "left_paren")
    compoundStr = compoundStr.replace(")", "right_paren")
    compoundStr = compoundStr.replace("<", "less_")
    compoundStr = compoundStr.replace(">", "greater_")
    compoundStr = compoundStr.replace("\n", "line_")

    # print(compoundStr)
    return compoundStr


def learnRules(columns, AMatrix, yVector, dataFile, node, weightRegularization, weightFeature, timeoutSec, ruleType,
               level, assignList, m, solver, test_phase, returnDictionary):
    # declare temp files for maxsat queries
    tempDataFile = dataFile[:23] + "_" + str(node) + dataFile[23:]
    WCNFFile = tempDataFile[:-5] + "_maxsat_rule.wcnf"
    outputMaxSAT = tempDataFile[:-5] + "_out.txt"

    totalTime = 0
    startTime = time.time()
    xSize = len(AMatrix[0])

    # generate maxsat queries
    # if (ruleType == 'or'):
    #     if (not True):
    #         generate_WCNF_file_with_card_constraints_for_all__samples(AMatrix, yVector, xSize, WCNFFile, level,
    #                                                                   weightRegularization,
    #                                                                   weightFeature, assignList,
    #                                                                   m, test_phase)
    #     else:
    #         generate_WCNF_file_with_card_constraint_per_sample(AMatrix, yVector, xSize, WCNFFile, level,
    #                                                            weightRegularization,
    #                                                            weightFeature, assignList,
    #                                                            m, test_phase)

    generateWCNFFile(AMatrix, yVector, weightRegularization, weightFeature, xSize, level, WCNFFile, assignList, 0,
                     test_phase)

    endTime=time.time()
    totalTime += endTime - startTime
    if (verbose):
        print("Time taken to model:" + str(endTime - startTime))

    # call a maxsat solver
    if (solver == "open-wbo"):
        if (verbose):
            print("using open wbo solver")
        cmd='open-wbo   ' + WCNFFile + ' > ' + outputMaxSAT
    if (solver == "maxHS"):
        cmd='maxhs -printBstSoln -cpu-lim=' + str(
            timeoutSec) + ' ' + WCNFFile + ' > ' + outputMaxSAT

    startTime = time.time()
    os.system(cmd)

    cmd = "rm " + WCNFFile
    os.system(cmd)
    endTime = time.time()
    totalTime += endTime - startTime
    if (verbose):
        print("Time taken to find the solution:" + str(endTime - startTime))

    # process solution to find significant columns and then generate rule
    f = open(outputMaxSAT, 'r')
    lines = f.readlines()
    f.close()
    optimumFound = False
    bestSolutionFound = False
    solution = ''
    for line in lines:
        if (line.strip().startswith('v') and optimumFound):
            solution = line.strip().strip('v ')
            break
        if (line.strip().startswith('c ') and bestSolutionFound):
            solution = line.strip().strip('c ')
            break
        if (line.strip().startswith('s OPTIMUM FOUND')):
            optimumFound = True
            if (verbose):
                print("Optimum solution found")
        if (line.strip().startswith('c Best Model Found:')):
            bestSolutionFound = True
            if (verbose):
                print("Best solution found")
    fields = solution.split()
    trueRules = []
    trueErrors = []
    zeroOneSolution = []
    for field in fields:
        if (int(field) > 0):
            zeroOneSolution.append(1.0)
        else:
            zeroOneSolution.append(0.0)
        if (int(field) > 0):

            if (abs(int(field)) <= level * xSize):
                trueRules.append(field)

            elif (abs(int(field)) <= level * xSize + len(yVector)):
                trueErrors.append(field)
    if (verbose):
        print("Cost of the best rule:" + str(weightRegularization *
                                             len(trueErrors) + weightFeature * len(trueRules)))
        print("The number of True Rule are:" + str(len(trueRules)))
        print("The number of errors are: " +
              str(len(trueErrors)) + " out of " + str(len(yVector)))

    # print("True Error are " + str([int(trueErrors[i]) for i in range(len(trueErrors))]))
    # for i in range(len(yVector)):
    #     for j in range(len(trueRules)):
    #         print(trueRules[j], end=' ')
    #         print(AMatrix[i][int(trueRules[j]) - 1], end=' '),
    #     print(yVector[i], end=' '),
    #     print(str(fields[i + xSize * level]))

    # if (verbose):
    #     for each_level in range(level):
    #         print(each_level)
    #         for i in range(len(yVector)):
    #             for j in range(len(trueRules)):
    #                 if (int(int(trueRules[j]) / xSize) == each_level):
    #                     # print(trueRules[j], end=' ')
    #                     print(AMatrix[i][int(trueRules[j]) - each_level * xSize - 1], end=' '),
    #             print(yVector[i], end=' '),
    #             print(str(fields[i + xSize * level]))
    xhat = []
    for i in range(level):
        xhat.append(numpy.array(zeroOneSolution[i * xSize:(i + 1) * xSize]))
    err = numpy.array(zeroOneSolution[xSize: xSize + len(yVector)])

    # recover rules
    recoveredRule = recoverRule(columns, xhat, ruleType, level, m)

    cmd = "rm " + outputMaxSAT
    os.system(cmd)

    returnDictionary[0] = xhat, err, fields[:level * xSize], len(trueErrors), len(trueRules), len(
        yVector), totalTime, recoveredRule


def partitionWithEqualProbability(X, y, partition_count):
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


def runTool():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset")
    parser.add_argument("--level", type=int, help="no of clauses", default=1)
    parser.add_argument("--lamda", type=int,
                        help="weightRegularization", default=10)
    parser.add_argument("--m", type=int, help="m", default=1)
    parser.add_argument("--node", type=int, help="node", default=0)
    parser.add_argument("--partition", type=int, help="partition", default=4)
    parser.add_argument("--c", type=int, help="weightFeature", default=1)
    parser.add_argument("--timeout", type=int,
                        help="timeout in seconds", default=1000)
    parser.add_argument("--runIndex", type=int, help="runIndex", default=0)
    parser.add_argument("--solver", type=str,
                        help="solver", default="open-wbo")

    # print(list(itertools.permutations([i for i in range(partition)])))
    # exit(0)

    args = parser.parse_args()
    partition = args.partition
    level = args.level
    m = args.m
    dataset = args.dataset
    weightRegularization = args.lamda
    weightFeature = args.c
    timeoutSec = args.timeout - 500
    runIndex = args.runIndex
    solver = args.solver
    ruleType = 'or'

    dataFilePath = "../Benchmarks/tempfiles/"

    # initial configurations, to be commented out after final publications
    # os.system("mkdir ../Benchmarks/tempfiles")
    # os.system("python init.py --dataset="+str(dataset))

    # os.system("mkdir ../Benchmarks/tempfiles_" + str(args.node))
    data = pickle.load(open("../Benchmarks/tempfiles/" +
                            dataset+"_"+str(runIndex)+".dat", "rb"))
    X_train = data['Xtrain']
    y_train = data['ytrain']
    X_test = data['Xtest']
    y_test = data['ytest']
    column_names = data['columns']

    dataFile = dataFilePath + dataset + "_train_" + str(runIndex) + ".data"

    # call method for making partitions
    ADfs, yVectors = partitionWithEqualProbability(X_train, y_train, partition)

    # train only on one partition
    completeCount = 0
    assignList = []
    totalTrainError = 0
    totalTrainTime = 0
    totalTestTime = 0
    totalTestError = 0
    ruleSize = 0
    totalTrainSampleCount = 0

    for partition_count in range(partition):

        manager = multiprocessing.Manager()
        returnDictionary = manager.dict()

        p = multiprocessing.Process(target=learnRules,
                                    args=(column_names, ADfs[partition_count],
                                          yVectors[partition_count],
                                          dataFile,
                                          args.node,
                                          weightRegularization,
                                          weightFeature,
                                          timeoutSec,
                                          ruleType,
                                          level,
                                          assignList,
                                          m,
                                          solver, False,
                                          returnDictionary))

        # execute program and put a cutoff time
        p.start()
        p.join(timeoutSec)

        if p.is_alive():
            if (verbose):
                print("running... let's kill it...")

            p.terminate()
            p.join()

        else:
            completeCount += 1
            xhat, err, assignList, error, ind_ruleSize, \
                trainSampleCount, train_time, \
                compoundRule = returnDictionary[0]

            if (verbose):
                compoundRule = compoundRule.replace("line_", "\n")
                compoundRule = compoundRule.replace("greater_", ">")
                compoundRule = compoundRule.replace("less_", "<")
                compoundRule = compoundRule.replace("right_paren", ")")
                compoundRule = compoundRule.replace("left_paren", "(")
                compoundRule = compoundRule.replace("^^", " ")

                print("\nPartition: " + str(partition_count + 1))
                print("\nrule->")
                print(compoundRule)
                print("\n")

            totalTrainSampleCount += trainSampleCount
            totalTrainTime += train_time
            totalTrainError += error

    if (completeCount == 0):
        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + str("Approx") + " " \
              + str(dataset) + " " \
              + str(level) + " " \
              + str(weightRegularization) + " " \
              + str(solver) + " " \
              + str(ruleSize) + " " \
              + str("NAN") + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(runIndex) + " " \
              + str(m) + " " + str(ruleType) + " 1 NAN"

        os.system(cmd)

    else:
        dataFile = dataFilePath + dataset + "_test_" + str(runIndex) + ".data"

        if (verbose):
            print("\n\n" + dataFile + "\n\n\n")

        returnDictionary = manager.dict()

        p = multiprocessing.Process(target=learnRules,
                                    args=(column_names, X_test, y_test,
                                          dataFile,
                                          args.node,
                                          weightRegularization,
                                          weightFeature,
                                          timeoutSec,
                                          ruleType,
                                          level,
                                          assignList, m,
                                          solver, True, returnDictionary))

        p.start()
        p.join(timeoutSec)

        if p.is_alive():
            if (verbose):
                print("running... let's kill it...")

            p.terminate()
            p.join()

            end_time = time.time()
        else:
            xhat, err, fields, error, ind_rule_size, \
                totalTestSampleCount, test_time, \
                compoundRule = returnDictionary[0]
            totalTestTime += test_time
            totalTestError += error
            ruleSize += ind_rule_size

            # save results
            cmd = "python ../output/dump_result.py ../output/result.csv " \
                  + str("Approx") + " " \
                  + str(dataset) + " " \
                  + str(level) + " " \
                  + str(weightRegularization) + " " \
                  + str(solver) + " " \
                  + str(ruleSize) + " " \
                  + str(partition) + " " \
                  + str(totalTrainTime) + " " \
                  + str((1.0 - (float(totalTrainError) / float(totalTrainSampleCount))) * 100.0) + " " \
                  + str(totalTestTime) + " " \
                  + str((1.0 - (float(totalTestError) / float(totalTestSampleCount))) * 100.0) + " " \
                  + str(runIndex) + " " \
                  + str(m) + " " + str(ruleType) + " 1  " + str(compoundRule)

            os.system(cmd)

            if (verbose):
                print("total time needed  train : " + str(totalTrainTime))

            if (verbose):
                print("Accuracy is:                 " + str(
                    (1.0 - (float(totalTrainError) / float(totalTrainSampleCount))) * 100.0))

            if (verbose):
                print(totalTrainSampleCount)

            if (verbose):
                print(
                    "Accuracy is:                 " + str(
                        (1.0 - (float(totalTestError) / float(totalTestSampleCount))) * 100.0))

            if (verbose):
                print(totalTestSampleCount)
                print("total time needed  : " + str(totalTestTime))

            if (verbose):
                compoundRule = compoundRule.replace("line_", "\n")
                compoundRule = compoundRule.replace("greater_", ">")
                compoundRule = compoundRule.replace("less_", "<")
                compoundRule = compoundRule.replace("right_paren", ")")
                compoundRule = compoundRule.replace("left_paren", "(")
                compoundRule = compoundRule.replace("^^", " ")

                print("dataset:                  " + dataset)
                print("no of clauses:            " + str(level))
                print("rule type:                " + ruleType)
                print("regularization parameter: " + str(weightRegularization))
                print("solver:                   " + solver)
                # print("partitions:               " + str(partition_count))
                print("train time:               " + str(totalTrainTime))
                print("test time:                " + str(totalTestTime))
                print("train accuracy:           " + str(
                    (1.0 - (float(totalTrainError) / float(totalTrainSampleCount))) * 100.0))
                print("test accuracy:            " + str(
                    (1.0 - (float(totalTestError) / float(totalTestSampleCount))) * 100.0))
                # print("test accuracy:            " + str((1.0 - (float(hold_error) / float(total_batch_size_hold))) * 100.0))

                print("\nrule->")
                print(compoundRule)
                print("\n")

    # os.system("rm -R  ../Benchmarks/tempfiles_" + str(args.node))


if __name__ == '__main__':
    runTool()
