import time
import sys
import os
import argparse
import pickle
import numpy
from subprocess import STDOUT, check_output
import multiprocessing

verbose = not True


def parseFiles(dataFile):
    # groupList= list of columns those represents same features in original data
    # groupMap =  converted column to original column

    x = pickle.load(open(dataFile, "rb"))
    AMatrix = x['A']
    yVector = x['y']
    ADf = x['A_df']

    groupList = []
    groupMap = {}
    currentIndex = 0
    listPosition = 0
    groupList.append([])
    previousElement = x['col_to_feat'][0]
    currentGroupIndex = 0
    for i in x['col_to_feat']:
        listPosition += 1
        if (not (i == previousElement)):
            currentGroupIndex += 1
        if (not (i == previousElement)):
            currentIndex += 1
            currElement = i
            groupList.append([])
            previousElement = i
        groupMap[listPosition] = currentGroupIndex
        groupList[currentIndex].append(listPosition)
    return ADf, AMatrix, yVector, groupList, groupMap, len(AMatrix[0])


def learnSoftClauses(xSize, ySize, level, weightFeature, weightRegularization,
                     assignList, test_phase):
    cnfClauses = ''
    numClauses = 0

    if (test_phase):
        topWeight = weightRegularization * ySize + 1 + weightFeature * xSize * level
        numClauses = 0
        for i in range(1, level * xSize + 1):
            numClauses += 1
            cnfClauses += str(weightFeature) + ' ' + str(-i) + ' 0\n'
        for i in range(level * xSize + 1, level * xSize + ySize + 1):
            numClauses += 1
            cnfClauses += str(weightRegularization) + ' ' + str(-i) + ' 0\n'
        for each_assign in assignList:
            numClauses += 1
            cnfClauses += str(topWeight) + ' ' + each_assign + ' 0\n'
    else:
        isEmptyAssignList = True
        total_additional_weight = 0;
        positiveLiteralWeight = weightFeature
        for each_assign in assignList:
            isEmptyAssignList = False
            numClauses += 1
            if (int(each_assign) > 0):

                cnfClauses += str(positiveLiteralWeight) + ' ' + each_assign + ' 0\n'
                total_additional_weight += positiveLiteralWeight

            else:
                cnfClauses += str(weightFeature) + ' ' + each_assign + ' 0\n'
                total_additional_weight += weightFeature

        # noise variables are to be kept consisitent
        for i in range(level * xSize + 1,
                       level * xSize + ySize + 1):
            numClauses += 1
            cnfClauses += str(weightRegularization) + ' ' + str(-i) + ' 0\n'

        # for the first step
        if (isEmptyAssignList):
            for i in range(1, level * xSize + 1):
                numClauses += 1
                cnfClauses += str(weightFeature) + ' ' + str(-i) + ' 0\n'
                total_additional_weight += weightFeature

        topWeight = int(weightRegularization * ySize + 1 + total_additional_weight)

    return topWeight, numClauses, cnfClauses


def generate_WCNF_file_with_card_constraint_per_sample(AMatrix, yVector, xSize, wCNFFileName, level,
                                                       weightRegularization,
                                                       weightFeature, assignList, m, test_phase):

    weight, constraintNumber, cnfClause = learnSoftClauses(xSize, len(yVector), level,
                                                           weightFeature,
                                                           weightRegularization,
                                                           assignList, test_phase)
    numOfVariables = level * xSize + len(yVector)

    for i in range(len(yVector)):
        # print(yVector[i])
        tempCnfClause = str(weight) + "  " + str(level * xSize + i + 1)

        for eachLevel in range(level):

            pbStr = ''
            if (yVector[i] == 1):
                tempPbStr = pbStr
                literalsCount = 1
                for j in range(len(AMatrix[i])):
                    if (AMatrix[i][j] == 1):
                        literalsCount = literalsCount + 1
                        pbStr += " +1 x" + str(xSize * eachLevel + j + 1)
                pbStr += " >=" + str(m) + "\n"
                if (literalsCount < m):
                    print("anytime?")
                    pbStr = tempPbStr
                    pbStr += "+1 x" + str(level * xSize + i + 1) + " >=1"
            else:

                # tseiten encoding

                tempPbStr = pbStr
                literalsCount = 1
                for j in range(len(AMatrix[i])):
                    if (AMatrix[i][j] == 1):
                        literalsCount = literalsCount + 1
                        pbStr += " -1 x" + str(xSize * eachLevel + j + 1)
                pbStr += " >=" + str(-m + 1) + "\n"
                if (literalsCount < m):
                    print("anytime?")
                    pbStr = tempPbStr
                    pbStr += "+1 x" + str(level * xSize + i + 1) + " >=1"
            # if (test_phase):
            #     print(pbStr[:-1])
            tempPBFile = open(wCNFFileName[:-5] + ".pb", "w")
            tempOutFile = wCNFFileName[:-5] + "_for_maxhs.out"
            tempPBFileName = wCNFFileName[:-5] + ".pb"
            tempPBFile.write(
                "* #variable= " + str(numOfVariables) + " #constraint= " + str(1) + "\n*\n")
            tempPBFile.write(pbStr[:-1])
            tempPBFile.close()
            cmd = "pbencoder " + tempPBFileName + "  > " + tempOutFile
            os.system(cmd)

            # print(pbStr)
            readTempPBFile = open(tempOutFile, "r")
            line = readTempPBFile.readline()
            while (line):
                # print(line)
                if (line.startswith("p")):
                    splitLiterals = line.split()
                    numOfVariables = int(splitLiterals[2])
                    constraintNumber += int(splitLiterals[3])
                    line = readTempPBFile.readline()
                    continue
                if (yVector[i] == 1):
                    # print(str(weight) + " " + str(level * xSize + i + 1) + " " + line[:-1] + "\n")
                    cnfClause += str(weight) + " " + str(level * xSize + i + 1) + " " + line[:-1] + "\n"
                else:
                    # print(str(weight) + " -" + str(numOfVariables + 1) + " " + line[:-1] + "\n")
                    cnfClause += str(weight) + " -" + str(numOfVariables + 1) + " " + line[:-1] + "\n"
                line = readTempPBFile.readline()

            # append to  an incomplete cnf clause which will be added to the formula later
            if (yVector[i] == 0):
                tempCnfClause += " " + str(numOfVariables + 1)
                numOfVariables += 1

            cmd = "rm " + wCNFFileName[:-5] + ".pb"
            os.system(cmd)

            cmd = "rm " + wCNFFileName[:-5] + "_for_maxhs.out"
            os.system(cmd)
        if (yVector[i] == 0):
            tempCnfClause = tempCnfClause + " 0\n"
            # print(tempCnfClause)
            cnfClause += tempCnfClause
            constraintNumber += 1

    fWCNFFile = open(wCNFFileName, "w")
    fWCNFFile.write("p wcnf " + str(numOfVariables) + " " + str(constraintNumber) + " " + str(weight) + "\n")
    fWCNFFile.write(cnfClause[:-1])
    fWCNFFile.close()
    return


def generate_WCNF_file_with_card_constraints_for_all__samples(AMatrix, yVector, xSize, wCNFFileName, level,
                                                              weightRegularization,
                                                              weightFeature, assignList, m, test_phase):
    # get sample independent clauses
    weight, constraintNumber, cnfClause = learnSoftClauses(xSize, len(yVector), level,
                                                           weightFeature,
                                                           weightRegularization,
                                                           assignList, test_phase)
    numOfVariables = level * xSize + len(yVector)
    # not considered for m of n type of rule
    if (level > 1):
        print("not consistent for level>1")

    # write cardinality constraints for all samples at a time and convert to cnf
    # cardinality (PB) constraints for samples (intermediate stage, new variables are introduced, which is removed in parsing stage)
    pbStr = ''
    max_count_aux_variables = len(
        AMatrix[0])  # to seperate PB encodings for each sample, new propositional variable are introduced
    for l in range(level):
        for i in range(len(yVector)):
            # for positive labeled samples
            if (yVector[i] == 1):
                tempPbStr = pbStr
                literalsCount = 1
                for j in range(len(AMatrix[i])):
                    if (AMatrix[i][j] == 1):
                        literalsCount = literalsCount + 1
                        pbStr += " +1 x" + str(i * max_count_aux_variables + j + 1)
                pbStr += " >=" + str(m) + "\n"
                if (literalsCount < m):
                    # if literal count in LHS is less than m, this sample is already a noise, so removed
                    print("need change ")
                    pbStr = tempPbStr
                    pbStr += "+1 x" + str(numOfVariables - len(yVector) + i + 1) + " >=1"
            # for negative labeled samples
            else:
                tempPbStr = pbStr
                literalsCount = 0
                for j in range(len(AMatrix[i])):
                    if (AMatrix[i][j] == 1):
                        literalsCount = literalsCount + 1
                        pbStr += " +1 ~x" + str(i * max_count_aux_variables + j + 1)
                pbStr += " >=" + str(literalsCount - m + 1) + "\n"
                if (literalsCount < m):
                    print("Need Change*********************** change literal count toooooooooooooooooooooooooooooo mst")
                    pbStr = tempPbStr
                    pbStr += "+1 x" + str(numOfVariables - len(yVector) + i + 1) + " >=1"

    # write PB constraints in file and pass to PBEncoder
    tempPBFile = open(wCNFFileName[:-5] + ".pb", "w")
    tempOutFile = wCNFFileName[:-5] + "_for_maxhs.out"
    tempPBFileName = wCNFFileName[:-5] + ".pb"

    max_count_aux_variables = max_count_aux_variables * len(yVector)  # assuming level=1
    tempPBFile.write(
        "* #variable= " + str(max_count_aux_variables) + " #constraint= " + str(
            len(yVector)) + "\n*\n")
    tempPBFile.write(pbStr[:-1])
    tempPBFile.close()
    cmd = "pbencoder " + tempPBFileName + "  > " + tempOutFile
    os.system(cmd)

    # parse output  of PBEncoder and construct maxsat constraint

    readTempPBFile = open(tempOutFile, "r")
    line = readTempPBFile.readline()
    noise_literal = xSize  # xSize

    aux_literal_count_reduce = xSize * len(yVector) - xSize - len(yVector)
    max_aux_var = 0
    while (line):

        noise_literal_change = False
        # get metadata of result i.e., no of variables, total number of constraints,
        if (line.startswith("p")):
            splitLiterals = line.split()
            numOfVariables = int(splitLiterals[2])
            constraintNumber += int(splitLiterals[3])
            line = readTempPBFile.readline()
            continue
        splitLiterals = line.split()
        revised_literals = ''

        critical_visit = True
        for literal in splitLiterals:
            abs_literal = abs(int(literal))
            # get 0 at the end of each line
            if (abs_literal == 0):
                revised_literals += literal + " "
            # used propositional variable
            elif (abs_literal < max_count_aux_variables):
                if (abs_literal == 0):
                    print("problem !!!!!!!!!!!!!")
                noise_literal = int((abs_literal - 1) / xSize) + xSize + 1
                noise_literal_change = True
                if (int(literal) > 0):
                    revised_literals += str((abs_literal - 1) % xSize + 1) + " "
                else:
                    revised_literals += "-" + str((abs_literal - 1) % xSize + 1) + " "
            # auxiliary variables
            else:
                new_aux_literal = abs_literal - aux_literal_count_reduce
                if (int(literal) > 0):
                    revised_literals += str(new_aux_literal) + " "
                else:
                    revised_literals += "-" + str(new_aux_literal) + " "
                if (abs_literal > max_aux_var):
                    max_aux_var = abs_literal
                    if (not noise_literal_change and critical_visit):
                        noise_literal += 1
                critical_visit = False

        cnfClause += str(weight) + " " + str(noise_literal) + " " + revised_literals + "\n"
        line = readTempPBFile.readline()
    readTempPBFile.close()

    fWCNFFile = open(wCNFFileName, "w")
    fWCNFFile.write(
        "p wcnf " + str(numOfVariables - aux_literal_count_reduce) + " " + str(constraintNumber) + " " + str(
            weight) + "\n")
    fWCNFFile.write(cnfClause[:-1])
    fWCNFFile.close()

    return


def learnRules(dataFile, node, weightRegularization, weightFeature, timeoutSec, ruleType, level,
               assignList, m, solver, test_phase, returnDictionary):
    tempdataFile = dataFile[:23] + "_" + str(node) + dataFile[23:]

    wCNFFileName = tempdataFile[:-5] + "_maxsat_rule.wcnf"
    outFileName = tempdataFile[:-5] + "_out.txt"

    total_time = 0
    startTime = time.time()
    ADf, AMatrix, yVector, groupList, groupMap, xSize = parseFiles(dataFile)
    yVector = yVector.values  # taking only the values in a matrix form

    endTime = time.time()
    total_time += endTime - startTime
    if (verbose):
        print("Time taken to parse:" + str(endTime - startTime))
    startTime = time.time()
    if (ruleType == 'or'):
        if (not False):
            generate_WCNF_file_with_card_constraint_per_sample(AMatrix, yVector, xSize, wCNFFileName, level,
                                                               weightRegularization, weightFeature, assignList,
                                                               m, test_phase)
        else:
            generate_WCNF_file_with_card_constraints_for_all__samples(AMatrix, yVector, xSize, wCNFFileName, level,
                                                                      weightRegularization,
                                                                      weightFeature, assignList,
                                                                      m, test_phase)

    # (AMatrix, yVector, xSize, wCNFFileName, level, weightRegularization, weightFeature, assignList, m)
    endTime = time.time()
    total_time += endTime - startTime
    if (verbose):
        print("Time taken to model:" + str(endTime - startTime))
    # cmd = 'open-wbo_release '+wCNFFileName+' > '+outFileName
    # print(wCNFFileName)
    if (solver == "open-wbo"):
        if (verbose):
            print("using open wbo solver")
        cmd = 'open-wbo   ' + wCNFFileName + ' > ' + outFileName
    if (solver == "maxHS"):
        cmd = 'maxhs -printBstSoln -cpu-lim=' + str(
            timeoutSec) + ' ' + wCNFFileName + ' > ' + outFileName
    # print(cmd)
    # cmd = 'LMHS '+wCNFFileName+' > '+outFileName
    # command = ['open-wbo_release', wCNFFileName, ' > ',outFileName]
    # command =['maxhs', '-printBstSoln', '-cpu-lim='+str(timeoutSec), wCNFFileName,' > ',outFileName]
    startTime = time.time()
    os.system(cmd)

    cmd = "rm " + wCNFFileName
    os.system(cmd)

    # output = check_output(command, stderr=STDOUT, timeout=timeoutSec+20)
    endTime = time.time()
    total_time += endTime - startTime
    if (verbose):
        print("Time taken to find the solution:" + str(endTime - startTime))
    f = open(outFileName, 'r')
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
    TrueErrors = []
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
                TrueErrors.append(field)

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

    if (verbose):
        print("Cost of the best rule:" + str(weightRegularization * len(TrueErrors) + weightFeature * len(trueRules)))
        print("The number of True Rule are:" + str(len(trueRules)))
        print("The number of errors are: " + str(len(TrueErrors)) + " out of " + str(len(yVector)))

    # print("True Error are " + str([int(TrueErrors[i])  for i in range(len(TrueErrors))]))
    # for i in range(len(yVector)):
    #     for j in range(len(trueRules)):
    #         print(trueRules[j], end=' ')
    #         print(AMatrix[i][int(trueRules[j]) - 1], end=' '),
    #     print(yVector[i], end=' '),
    #     print(str(fields[i + xSize * level]))

    xhat = []
    for i in range(level):
        # print(i * xSize,(i + 1) * xSize)
        xhat.append(numpy.array(zeroOneSolution[i * xSize:(i + 1) * xSize]))
    err = numpy.array(zeroOneSolution[xSize: xSize + len(yVector)])
    rule_str_rec = recoverRule(ADf, xhat, ruleType, level, m)
    # if (verbose):
    #     print("PRINTING RULE")
    #     print(rule_str_rec)

    cmd = "rm " + outFileName
    os.system(cmd)

    returnDictionary[0] = xhat, err, fields[:level * xSize], len(TrueErrors), len(trueRules), len(
        yVector), total_time, rule_str_rec


def runTool():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset")
    parser.add_argument("--level", type=int, help="no of clauses", default=1)
    parser.add_argument("--lamda", type=int, help="weightRegularization", default=10)
    parser.add_argument("--m", type=int, help="m", default=2)
    parser.add_argument("--node", type=int, help="node", default=0)
    parser.add_argument("--c", type=int, help="weightFeature", default=1)
    # parser.add_argument("--lambda", type=int, help="lambda", default=10)
    parser.add_argument("--timeout", type=int, help="timeout in seconds", default=1000)
    parser.add_argument("--runIndex", type=int, help="runIndex", default=0)
    parser.add_argument("--solver", type=str, help="solver", default="open-wbo")

    args = parser.parse_args()
    level = args.level
    m = args.m
    dataset = args.dataset
    weightRegularization = args.lamda
    weightFeature = args.c
    timeoutSec = args.timeout
    runIndex = args.runIndex
    solver = args.solver
    ruleType = 'or'

    dataFielPath = "../Benchmarks/tempfiles/"

    # initial configurations, to be commented out after final publications
    # os.system("mkdir ../Benchmarks/tempfiles")
    # os.system("python init.py --dataset="+str(dataset))

    # os.system("mkdir ../Benchmarks/tempfiles_" + str(args.node))

    totalTrainError = 0
    totalTrainTime = 0
    totalTestTime = 0
    totalTestError = 0
    ruleSize = 0

    assignList = []

    dataFile = dataFielPath + dataset + "_train_" + str(runIndex) + ".data"
    if (verbose):
        print("\n\n" + dataFile + "\n\n\n")
    completeCount = 0
    manager = multiprocessing.Manager()
    returnDictionary = manager.dict()

    p = multiprocessing.Process(target=learnRules,
                                args=(dataFile,
                                      args.node,
                                      weightRegularization,
                                      weightFeature,
                                      timeoutSec,
                                      ruleType, level,
                                      assignList, m,
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

        xhat, err, fields, error, individualRuleSize, totalTrainSampleCount, time, compoundRule = returnDictionary[0]
        totalTrainTime += time
        totalTrainError += error
        assignList = fields

    if (verbose):
        print("\n\n" + dataFile + "\n\n\n")
    if (completeCount == 0):

        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + str("Exact") + " " \
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
              + str(m) + " " + str(ruleType) + " NAN"

        os.system(cmd)

    else:
        returnDictionary = manager.dict()

        dataFile = dataFielPath + dataset + "_test_" + str(runIndex) + ".data"

        p = multiprocessing.Process(target=learnRules,
                                    args=(dataFile,
                                          args.node,
                                          weightRegularization,
                                          weightFeature,
                                          timeoutSec,
                                          ruleType, level,
                                          assignList, m,
                                          solver, True,
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

            xhat, err, fields, error, individualRuleSize, totalTestSampleCount, time, compoundRule = returnDictionary[0]

            totalTestTime += time
            totalTestError += error
            ruleSize += individualRuleSize

            cmd = "python ../output/dump_result.py ../output/result.csv " \
                  + str("Exact") + " " \
                  + str(dataset) + " " \
                  + str(level) + " " \
                  + str(weightRegularization) + " " \
                  + str(solver) + " " \
                  + str(ruleSize) + " " \
                  + str("NAN") + " " \
                  + str(totalTrainTime) + " " \
                  + str((1.0 - (float(totalTrainError) / float(totalTrainSampleCount))) * 100.0) + " " \
                  + str(totalTestTime) + " " \
                  + str((1.0 - (float(totalTestError) / float(totalTestSampleCount))) * 100.0) + " " \
                  + str(runIndex) + " " \
                  + str(m) + " " + str(ruleType) + "  " + str(compoundRule)

            os.system(cmd)

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


def recoverRule(ADf, x_hat_vec, ruleType, level, m):
    compoundStr = '( '
    for i in range(level):
        x_hat = x_hat_vec[i]
        inds_nnz = numpy.where(abs(x_hat) > 1e-4)[0]
        str_clauses = [' '.join(ADf.columns[ind]) for ind in inds_nnz]
        # rule_sep = ' %s ' % ruleType
        rule_sep = ' + '

        rule_str = rule_sep.join(str_clauses)
        if (ruleType == 'and'):
            rule_str = rule_str.replace('<=', '??').replace('>', '<=').replace('??', '>')
        compoundStr += rule_str
        compoundStr += ' )>=' + str(m)

        if (i < level - 1):
            if (ruleType == 'and'):
                compoundStr += ' or\n( '
            if (ruleType == 'or'):
                compoundStr += ' and\n( '

    compoundStr = compoundStr.replace(" ", "^^")
    compoundStr = compoundStr.replace("(", "left_paren")
    compoundStr = compoundStr.replace(")", "right_paren")
    compoundStr = compoundStr.replace("<", "less_")
    compoundStr = compoundStr.replace(">", "greater_")
    compoundStr = compoundStr.replace("\n", "line_")

    return compoundStr


if __name__ == '__main__':
    runTool()
