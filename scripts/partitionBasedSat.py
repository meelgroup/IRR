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
    recoverRule, generate_WCNF_file_with_card_constraint_per_sample
from helper import partition_with_eq_prob

verbose = not True


def learnRules(ADf, AMatrix, yVector, dataFile, node, weightRegularization, weightFeature, timeoutSec, ruleType,
               level, assignList, m, solver, test_phase, returnDictionary):
    # declare temp files for maxsat queries
    tempDataFile = dataFile[:23] + "_" + str(node) + dataFile[23:]
    WCNFFile = tempDataFile[:-5] + "_maxsat_rule.wcnf"
    outputMaxSAT = tempDataFile[:-5] + "_out.txt"

    totalTime = 0
    startTime = time.time()
    rows, xSize = AMatrix.shape

    # generate maxsat queries
    if (ruleType == 'or'):
        if (not True):
            generate_WCNF_file_with_card_constraints_for_all__samples(AMatrix, yVector, xSize, WCNFFile, level,
                                                                      weightRegularization,
                                                                      weightFeature, assignList,
                                                                      m, test_phase)
        else:
            generate_WCNF_file_with_card_constraint_per_sample(AMatrix, yVector, xSize, WCNFFile, level,
                                                               weightRegularization,
                                                               weightFeature, assignList,
                                                               m, test_phase)
    endTime = time.time()
    totalTime += endTime - startTime
    if (verbose):
        print("Time taken to model:" + str(endTime - startTime))

    # call a maxsat solver
    if (solver == "open-wbo"):
        if (verbose):
            print("using open wbo solver")
        cmd = 'open-wbo   ' + WCNFFile + ' > ' + outputMaxSAT
    if (solver == "maxHS"):
        cmd = 'maxhs -printBstSoln -cpu-lim=' + str(
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
        print("Cost of the best rule:" + str(weightRegularization * len(trueErrors) + weightFeature * len(trueRules)))
        print("The number of True Rule are:" + str(len(trueRules)))
        print("The number of errors are: " + str(len(trueErrors)) + " out of " + str(len(yVector)))

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
    recoveredRule = recoverRule(ADf, xhat, ruleType, level, m)

    cmd = "rm " + outputMaxSAT
    os.system(cmd)

    returnDictionary[0] = xhat, err, fields[:level * xSize], len(trueErrors), len(trueRules), len(
        yVector), totalTime, recoveredRule


def runTool():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset")
    parser.add_argument("--level", type=int, help="no of clauses", default=1)
    parser.add_argument("--lamda", type=int, help="weightRegularization", default=10)
    parser.add_argument("--m", type=int, help="m", default=1)
    parser.add_argument("--node", type=int, help="node", default=0)
    parser.add_argument("--partition", type=int, help="partition", default=4)
    parser.add_argument("--c", type=int, help="weightFeature", default=1)
    parser.add_argument("--timeout", type=int, help="timeout in seconds", default=1000)
    parser.add_argument("--runIndex", type=int, help="runIndex", default=0)
    parser.add_argument("--solver", type=str, help="solver", default="open-wbo")

    # print(list(itertools.permutations([i for i in range(partition)])))
    # exit(0)

    args = parser.parse_args()
    partition = args.partition
    level = args.level
    m = args.m
    dataset = args.dataset
    weightRegularization = args.lamda
    weightFeature = args.c
    timeoutSec = args.timeout
    runIndex = args.runIndex
    solver = args.solver
    ruleType = 'or'

    dataFilePath = "../Benchmarks/tempfiles/"

    # initial configurations, to be commented out after final publications
    # os.system("mkdir ../Benchmarks/tempfiles")
    # os.system("python init.py --dataset="+str(dataset))

    # os.system("mkdir ../Benchmarks/tempfiles_" + str(args.node))

    dataFile = dataFilePath + dataset + "_train_" + str(runIndex) + ".data"
    if (verbose):
        print("\n\n" + dataFile + "\n\n\n")
    # make partition and learn for each partition and
    # retrieve the data at first for making partition

    ADf, AMatrix, yVector, groupList, groupMap, xSize = parseFiles(dataFile)

    # call method for making partitions
    ADfs, yVectors = partition_with_eq_prob(ADf, yVector, partition)

    # train only on one partition
    completeCount = 0
    for p in range(partition):

        # train on all partitions
        if (p > 0):
            break

        # iterate over sequence of all partition sequences

        sequences = list(itertools.permutations([i for i in range(partition)]))

        for sequence in sequences:
            totalTrainError = 0
            totalTrainTime = 0
            totalTestTime = 0
            totalTestError = 0
            ruleSize = 0
            totalTrainSampleCount = 0

            assignList = []

            for i in (list(sequence)):

                # train on p-th partition only
                # if(i!=p):
                #     continue
                #

                manager = multiprocessing.Manager()
                returnDictionary = manager.dict()

                p = multiprocessing.Process(target=learnRules,
                                            args=(ADfs[i],
                                                  ADfs[i].as_matrix(),
                                                  yVectors[i].values,
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

                        print("\nPartition: " + str(i + 1))
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
                ADf, AMatrix, yVector, groupList, groupMap, xSize = parseFiles(dataFile)

                if (verbose):
                    print("\n\n" + dataFile + "\n\n\n")

                returnDictionary = manager.dict()

                p = multiprocessing.Process(target=learnRules,
                                            args=(ADf, AMatrix,
                                                  yVector.values,
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

            # execute only the first sequence
            break
            #

    # os.system("rm -R  ../Benchmarks/tempfiles_" + str(args.node))


if __name__ == '__main__':
    runTool()
