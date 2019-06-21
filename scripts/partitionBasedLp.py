from scipy.optimize import linprog
import time
import argparse
import pickle
import multiprocessing
from pulp import *
import numpy as np
import math
from sklearn.model_selection import train_test_split

# from cp_solver import call_cp_solver, call_multilevel_cp_solver
# from helper import partition_with_eq_prob
from lp import ParseFiles, retrive_rule_and_train_acc, find_test_acc, debug

verbose = not True
learn_eta = False
memlimit = 1000*4


def call_cplex(A, y, eta, no_features, no_samples, weight_feature,
               weight_regularization, level, solution,
               solver,
               eta_clause, learn_eta_clause, timelimit, node):
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)

    # Establish the Linear Programming Model
    myProblem = cplex.Cplex()

    feature_variable = []
    variable_list = []
    objective_coefficient = []
    variable_count = 0

    for eachLevel in range(level):
        for i in range(no_features):
            feature_variable.append("b_" + str(i + 1) + str("_") + str(eachLevel + 1))

    variable_list = variable_list + feature_variable

    slack_variable = []
    for i in range(no_samples):
        slack_variable.append("s_" + str(i + 1))

    variable_list = variable_list + slack_variable

    if (learn_eta_clause):
        variable_list.append("eta_clause")

    if (learn_eta):
        variable_list.append("eta_clit")

    for i in range(len(y)):
        for eachLevel in range(level):
            variable_list.append("ax_" + str(i + 1) + str("_") + str(eachLevel + 1))

    myProblem.variables.add(names=variable_list)

    # if (len(solution) == 0):
    #     for eachLevel in range(level):
    #         for i in range(no_features):
    #             feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
    #             objective.append(weight_feature * feature_variable[eachLevel * no_features + i])
    # else:
    #     for eachLevel in range(level):
    #         for i in range(no_features):
    #             feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
    #             if (solution[eachLevel * no_features + 1] > 0):
    #                 objective.append(-weight_feature * feature_variable[eachLevel * no_features + i])
    #             else:
    #                 objective.append(weight_feature * feature_variable[eachLevel * no_features + i])

    if (len(solution) == 0):
        for eachLevel in range(level):
            for i in range(no_features):
                if (solution[eachLevel * no_features + 1] > 0):
                    objective_coefficient.append(-weight_feature)
                else:
                    objective_coefficient.append(weight_feature)

                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)
                myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
                myProblem.objective.set_linear([(variable_count, objective_coefficient[variable_count])])
                variable_count += 1
    else:
        for eachLevel in range(level):
            for i in range(no_features):
                objective_coefficient.append(weight_feature)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)
                myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
                myProblem.objective.set_linear([(variable_count, objective_coefficient[variable_count])])
                variable_count += 1

    # slack_variable = []
    for i in range(no_samples):
        objective_coefficient.append(weight_regularization)
        myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, 1)
        myProblem.objective.set_linear([(variable_count, objective_coefficient[variable_count])])
        variable_count += 1

    myProblem.objective.set_sense(myProblem.objective.sense.minimize)

    var_eta_clause = -1

    if (learn_eta_clause):
        myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, level)
        var_eta_clause = variable_count
        variable_count += 1

    var_eta_literal = -1
    constraint_count = 0

    if (learn_eta):

        myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, no_features)
        var_eta_literal = variable_count
        variable_count += 1

        for eachLevel in range(level):
            constraint = []

            for j in range(no_features):
                constraint.append(1)

            constraint.append(-1)

            myProblem.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind=[eachLevel * no_features + j for j in range(no_features)] + [var_eta_literal],
                                     val=constraint)],
                rhs=[0],
                names=["c" + str(constraint_count)],
                senses=["G"]
            )
            constraint_count += 1

    for i in range(len(y)):
        if (y[i] == 1):

            auxiliary_index = []

            for eachLevel in range(level):
                constraint = [feature for feature in A[i]]

                myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)

                constraint.append(no_features)

                auxiliary_index.append(variable_count)

                if (learn_eta):

                    constraint.append(-1)

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                             var_eta_literal],
                            val=constraint)],
                        rhs=[0],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count],
                            val=constraint)],
                        rhs=[eta],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                variable_count += 1

            if (learn_eta_clause):

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + level * no_features, var_eta_clause] + auxiliary_index,
                        # 1st slack variable = level * no_features
                        val=[level, -1] + [-1 for j in range(level)])],
                    rhs=[- level],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

            else:

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + level * no_features] + auxiliary_index,  # 1st slack variable = level * no_features
                        val=[level] + [-1 for j in range(level)])],
                    rhs=[- level + eta_clause],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

        else:

            auxiliary_index = []

            for eachLevel in range(level):
                constraint = [feature for feature in A[i]]
                myProblem.variables.set_types(variable_count, myProblem.variables.type.integer)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)

                constraint.append(- no_features)

                auxiliary_index.append(variable_count)

                if (learn_eta):

                    constraint.append(-1)

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                             var_eta_literal],
                            val=constraint)],
                        rhs=[-1],
                        names=["c" + str(constraint_count)],
                        senses=["L"]
                    )

                    constraint_count += 1
                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count],
                            val=constraint)],
                        rhs=[eta - 1],
                        names=["c" + str(constraint_count)],
                        senses=["L"]
                    )

                    constraint_count += 1

                variable_count += 1

            if (learn_eta_clause):

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + level * no_features, var_eta_clause] + auxiliary_index,
                        # 1st slack variable = level * no_features
                        val=[level, 1] + [-1 for j in range(level)])],
                    rhs=[1],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1


            else:

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + level * no_features] + auxiliary_index,  # 1st slack variable = level * no_features
                        val=[level] + [-1 for j in range(level)])],
                    rhs=[- eta_clause + 1],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

    # set parameters
    myProblem.parameters.clocktype.set(1)  # cpu time (exact time)
    myProblem.parameters.timelimit.set(timelimit)
    myProblem.parameters.workmem.set(memlimit)
    myProblem.set_log_stream(None)
    myProblem.set_error_stream(None)
    myProblem.set_warning_stream(None)
    myProblem.set_results_stream(None)
    myProblem.parameters.mip.tolerances.mipgap.set(.01)
    myProblem.parameters.mip.limits.treememory.set(memlimit)
    myProblem.parameters.workdir.set("../Benchmarks/tempfiles_" + str(node) + "/")
    myProblem.parameters.mip.strategy.file.set(2)
    myProblem.parameters.threads.set(1)

    # Solve the model and print the answer
    start_time = myProblem.get_time()
    start_det_time = myProblem.get_dettime()
    myProblem.solve()
    # solution.get_status() returns an integer code
    status = myProblem.solution.get_status()

    end_det_time = myProblem.get_dettime()

    end_time = myProblem.get_time()
    if (verbose):
        print("Total solve time (sec.):", end_time - start_time)
        print("Total solve time (sec.):", end_det_time - start_det_time)

        print("Solution status = ", status, ":")
        print(myProblem.solution.status[status])
        print("Objective value = ", myProblem.solution.get_objective_value())
        print(myProblem.solution.MIP.get_mip_relative_gap())

    solution = []
    for i in range(len(feature_variable)):
        solution.append(myProblem.solution.get_values(i))

    for i in range(len(slack_variable)):
        solution.append(myProblem.solution.get_values(level * no_features + i))

    if (learn_eta_clause and learn_eta):
        return solution, int(myProblem.solution.get_values(var_eta_literal)), int(
            myProblem.solution.get_values(var_eta_clause))

    elif (learn_eta_clause):
        return solution, eta, int(myProblem.solution.get_values(var_eta_clause))

    elif (learn_eta):
        return solution, int(myProblem.solution.get_values(var_eta_literal)), eta_clause
    else:
        return solution, eta, eta_clause
def partitionWithEqualProbability( X, y,partition_count):
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
    parser.add_argument("--timeout", type=int, help="cpu timeout in seconds", default=2000)
    parser.add_argument("--runIndex", type=int, help="runIndex", default=0)
    parser.add_argument("--m", type=int, help="m", default=2)
    parser.add_argument("--level", type=int, help="level", default=1)
    parser.add_argument("--c", type=int, help="weight of feature vector", default=1)
    parser.add_argument("--lamda", type=int, help="weight of regularization param", default=10)
    parser.add_argument("--node", type=int, help="node", default=0)
    parser.add_argument("--partition", type=int, help="partition", default=4)
    parser.add_argument("--solver", type=str, help="solver", default="cplex")
    parser.add_argument("--eta_clause", type=int, help="eta_clause", default=-1)

    # parser.add_argument("--classifier", type=str, help="Classifier name")

    args = parser.parse_args()
    partition = args.partition
    level = args.level
    dataset = args.dataset
    timeOut = args.timeout - 500
    runIndex = args.runIndex
    weight_feature = args.c
    weight_regularization = args.lamda
    m = args.m
    eta_clause = args.eta_clause

    learn_eta_clause = False
    if (eta_clause == -1):
        learn_eta_clause = True

    # os.system("mkdir ../Benchmarks/tempfiles_" + str(args.node))

    total_time_needed_train = 0
    overall_acc_train = 0
    total_time_needed_test = 0
    overall_acc_test = 0
    datafile_path = "../Benchmarks/tempfiles/"
    actual_complete = 0
    if (verbose):
        print(runIndex)

    data = pickle.load(open("../Benchmarks/tempfiles/" +
                            dataset+"_"+str(runIndex)+".dat", "rb"))
    X_train = data['Xtrain']
    y_train = data['ytrain']
    X_test = data['Xtest']
    y_test = data['ytest']
    column_names = data['columns']



    # call method for making partitions
    A_dfs, yVectors = partitionWithEqualProbability(X_train, y_train, partition)

    solution = []
    compound_rule = ""
    rule_size = 0
    # train on all partitions

    for i in range(partition):

        # column_names = [str(a) + str(" ") + str(b) + str(" ") + str(c) for (a, b, c) in
        #                 X_df.columns]

        start_time = time.time()
        X = A_dfs[i]
        Y = yVectors[i]

        if (args.solver != "cplex"):
            solution, eta, eta_clause = call_generalized_ilp_solver_clause_relaxation(X, list(Y), m, len(X[0]), len(Y),
                                                                                      weight_feature,
                                                                                      weight_regularization, solution,
                                                                                      level, args.solver,
                                                                                      eta_clause, learn_eta_clause)
        else:
            solution, eta, eta_clause = call_cplex(X, list(Y), m, len(X[0]), len(Y),
                                                   weight_feature,
                                                   weight_regularization,
                                                   level, args.solver, solution,
                                                   eta_clause, learn_eta_clause, int(timeOut / partition), args.node)

        actual_complete += 1

        # retrieve rules and calculate runtime, prediction error
        acc_train, compound_rule, rule_size = retrive_rule_and_train_acc(solution, len(X[0]), m, column_names,
                                                                         level, eta_clause)
        end_time = time.time()
        # if (verbose):
        #     debug(level, X, Y, solution)

        train_time = end_time - start_time
        total_time_needed_train += train_time
        overall_acc_train += acc_train

        if (verbose):
            compound_rule_partition = compound_rule
            compound_rule_partition = compound_rule_partition.replace("line_", "\n")
            compound_rule_partition = compound_rule_partition.replace("greater_", ">")
            compound_rule_partition = compound_rule_partition.replace("less_", "<")
            compound_rule_partition = compound_rule_partition.replace("right_paren", ")")
            compound_rule_partition = compound_rule_partition.replace("left_paren", "(")
            compound_rule_partition = compound_rule_partition.replace("^^", " ")

            print("\nPartition: " + str(i + 1))
            print("\nrule->")
            print(compound_rule_partition)
            print("\n")

    if (actual_complete == 0):
        total_time_needed_train = timeOut
        overall_acc_train = -1
        total_time_needed_test = timeOut
        overall_acc_test = -1

        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + str("Approx_LP") + " " \
              + str(dataset) + " " \
              + str(level) + " " \
              + str(weight_regularization) + " " \
              + str(args.solver) + " " \
              + str("NAN") + " " \
              + str(partition) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(runIndex) + " " \
              + str(m) + " " + str("or") + " " + str(-1) + " NAN"

        os.system(cmd)

    else:

        # calculate test error and computation time

        start_time = time.time()
        
        acc_test, solution_test = find_test_acc(solution, X_test, y_test, m, level, eta_clause)
        end_time = time.time()
        test_time = end_time - start_time
        total_time_needed_test += test_time
        overall_acc_test += acc_test

        if (verbose):
            print("test\n\n\n")
            # debug(level, X_test, y_test, solution_test)

        # normalize calculation, not necessary for  test set,
        total_time_needed_train = float(total_time_needed_train)
        overall_acc_train = float(overall_acc_train / actual_complete)

        # save result in files
        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + str("Approx_LP") + " " \
              + str(dataset) + " " \
              + str(level) + " " \
              + str(weight_regularization) + " " \
              + str(args.solver) + " " \
              + str(rule_size) + " " \
              + str(partition) + " " \
              + str(total_time_needed_train) + " " \
              + str(overall_acc_train * 100.0) + " " \
              + str(total_time_needed_test) + " " \
              + str(overall_acc_test * 100.0) + " " \
              + str(runIndex) + " " \
              + str(m) + " " + str("or") + " " + str(eta_clause) + "  " + str(compound_rule)
        # print(cmd)
        os.system(cmd)
        if (verbose):
            compound_rule = compound_rule.replace("line_", "\n")
            compound_rule = compound_rule.replace("greater_", ">")
            compound_rule = compound_rule.replace("less_", "<")
            compound_rule = compound_rule.replace("right_paren", ")")
            compound_rule = compound_rule.replace("left_paren", "(")
            compound_rule = compound_rule.replace("^^", " ")

            print("dataset:                  " + dataset)
            print("no of clauses:            " + str(level))
            # print("rule type:                " + rule_type)
            print("regularization parameter: " + str(weight_regularization))
            print("solver:                   " + args.solver)
            print("required clause:          " + str(eta_clause))
            print("required literals:        " + str(eta))
            print("train time:               " + str(total_time_needed_train))
            print("test time:                " + str(total_time_needed_test))
            print("train accuracy:           " + str(overall_acc_train * 100.0))
            print("test accuracy:            " + str(overall_acc_test * 100.0))
            # print("test accuracy:            " + str((1.0 - (float(hold_error) / float(total_batch_size_hold))) * 100.0))

            print("\nrule->")
            print(compound_rule)
            print("\n")

    # os.system("rm -R  ../Benchmarks/tempfiles_" + str(args.node))


def call_generalized_ilp_solver_clause_relaxation(A, y, eta, no_features, no_samples, weight_feature,
                                                  weight_regularization, solution, level,
                                                  solver,
                                                  eta_clause, learn_eta_clause):
    # print(learn_eta_clause)
    # todo (done) slack variable range changed, in constraints, eta is multiplied with slack variable to remain consistency
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)
    # print(y)

    prob = LpProblem("m_of_n", LpMinimize)

    # declare variables & objective
    objective = []
    feature_variable = []

    if (len(solution) == 0):
        for eachLevel in range(level):
            for i in range(no_features):
                feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                objective.append(weight_feature * feature_variable[eachLevel * no_features + i])
    else:
        for eachLevel in range(level):
            for i in range(no_features):
                feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                if (solution[eachLevel * no_features + 1] > 0):
                    objective.append(-weight_feature * feature_variable[eachLevel * no_features + i])
                else:
                    objective.append(weight_feature * feature_variable[eachLevel * no_features + i])
    slack_variable = []
    for i in range(no_samples):
        slack_variable.append(LpVariable("s_" + str(i + 1), 0, 1, LpInteger))
        objective.append(weight_regularization * slack_variable[i])

    # Objective
    prob += lpSum(objective)
    # for auxiliary variables
    auxiliary_variables = []

    # Constraints
    # learn only eta_clause
    if (learn_eta_clause):
        var_eta_clause = (LpVariable("eta_clause", 1, level, LpInteger))

    if (learn_eta):
        var_eta_literal = (LpVariable("eta_lit", 1, no_features, LpInteger))

        for eachLevel in range(level):
            constraint = []

            for j in range(no_features):
                constraint.append(feature_variable[eachLevel * no_features + j])

            prob += lpSum(constraint) >= var_eta_literal

    for i in range(len(y)):
        if (y[i] == 1):

            auxiliary_constraint = []

            for eachLevel in range(level):
                constraint = []
                for j in range(len(A[i])):
                    constraint.append(A[i][j] * feature_variable[eachLevel * no_features + j])

                auxiliary_variables.append(
                    LpVariable("ax_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))

                constraint.append(no_features * auxiliary_variables[-1])
                auxiliary_constraint.append(auxiliary_variables[-1])

                if (learn_eta):
                    prob += lpSum(constraint) >= var_eta_literal
                else:
                    prob += lpSum(constraint) >= eta
            if (learn_eta_clause):
                prob += level * slack_variable[i] + level - var_eta_clause >= lpSum(auxiliary_constraint)
            else:
                prob += level * slack_variable[i] + level - eta_clause >= lpSum(auxiliary_constraint)
        else:

            auxiliary_constraint = []
            for eachLevel in range(level):

                constraint = []

                for j in range(len(A[i])):
                    constraint.append(A[i][j] * feature_variable[eachLevel * no_features + j])

                auxiliary_variables.append(
                    LpVariable("ax_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                # attach the last aux variable
                constraint.append(-1 * no_features * auxiliary_variables[-1])
                auxiliary_constraint.append(auxiliary_variables[-1])
                # prob += lpSum(constraint) <= var_eta_literal - 1

                if (learn_eta):
                    prob += lpSum(constraint) <= var_eta_literal - 1
                else:
                    prob += lpSum(constraint) <= eta - 1
            # constraint for aux variables
            if (learn_eta_clause):
                prob += level * slack_variable[i] + var_eta_clause >= lpSum(auxiliary_constraint) + 1
            else:
                prob += level * slack_variable[i] + eta_clause >= lpSum(auxiliary_constraint) + 1
            # slack_variable[i].upBound = None
    # print(prob)
    # call tentative solver

    # print(len(auxiliary_variables))
    # select a solver
    if (solver == "default"):
        prob.solve()
    elif (solver == "gurobi"):
        prob.solve(GUROBI())
    elif (solver == "cplex"):
        prob.solve(CPLEX(msg=0))
    elif (solver == "glpk"):
        prob.solve(GLPK(msg=0))

    # parse solution solution
    solution = []
    # print(feature_variable)
    # print(prob)
    # print(feature_variable)
    for i in range(len(feature_variable)):
        solution.append(value(feature_variable[i]))

    for i in range(len(slack_variable)):
        solution.append(value(slack_variable[i]))

    # if (verbose):
    #     print("eta_lit: ", value(var_eta_literal))
    #     print("eta_cl: ", value(var_eta_clause))

    if (learn_eta_clause and learn_eta):
        return solution, int(value(var_eta_literal)), int(value(var_eta_clause))

    elif (learn_eta_clause):
        return solution, eta, int(value(var_eta_clause))

    elif (learn_eta):
        return solution, int(value(var_eta_literal)), eta_clause
    else:
        return solution, eta, eta_clause


# modifed for partition based learning
def call_generalized_ilp_solver(A, y, eta, no_features, no_samples, weight_feature, weight_regularization, solution,
                                level, solver,
                                return_dict):
    # todo (done) slack variable range changed, in constraints, eta is multiplied with slack variable to remain consistency
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)
    # print(y)

    prob = LpProblem("m_of_n", LpMinimize)

    # declare variables & objective
    objective = []
    feature_variable = []

    if (len(solution) == 0):
        for eachLevel in range(level):
            for i in range(no_features):
                feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                objective.append(weight_feature * feature_variable[eachLevel * no_features + i])
    else:
        for eachLevel in range(level):
            for i in range(no_features):
                feature_variable.append(LpVariable("b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                if (solution[eachLevel * no_features + 1] > 0):
                    objective.append(-weight_feature * feature_variable[eachLevel * no_features + i])
                else:
                    objective.append(weight_feature * feature_variable[eachLevel * no_features + i])
    slack_variable = []
    for i in range(no_samples):
        slack_variable.append(LpVariable("s_" + str(i + 1), 0, 1, LpInteger))
        objective.append(weight_regularization * slack_variable[i])

    # Objective
    prob += lpSum(objective)
    # for auxiliary variables
    auxiliary_variables = []

    # Constraints
    for i in range(len(y)):
        if (y[i] == 1):

            for eachLevel in range(level):
                constraint = []
                for j in range(len(A[i])):
                    constraint.append(A[i][j] * feature_variable[eachLevel * no_features + j])

                constraint.append(no_features * slack_variable[i])

                prob += lpSum(constraint) >= eta
        else:

            auxiliary_constraint = []
            for eachLevel in range(level):

                constraint = []

                for j in range(len(A[i])):
                    constraint.append(A[i][j] * feature_variable[eachLevel * no_features + j])

                auxiliary_variables.append(
                    LpVariable("ax_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                # attach the last aux variable
                constraint.append(-1 * no_features * auxiliary_variables[-1])
                auxiliary_constraint.append(auxiliary_variables[-1])
                prob += lpSum(constraint) <= eta - 1

            # constraint for aux variables
            prob += level * (1 + slack_variable[i]) >= lpSum(auxiliary_constraint) + 1
            # slack_variable[i].upBound = None
    # print(prob)
    # call tentative solver

    # select a solver
    if (solver == "default"):
        prob.solve()
    elif (solver == "gurobi"):
        prob.solve(GUROBI())
    elif (solver == "cplex"):
        prob.solve(CPLEX(msg=0))
    elif (solver == "glpk"):
        prob.solve(GLPK(msg=0, options=['--mipgap', '0.01']))
    # parse solution solution
    solution = []
    # print(feature_variable)
    # print(prob)
    # print(feature_variable)
    for i in range(len(feature_variable)):
        solution.append(value(feature_variable[i]))

    for i in range(len(slack_variable)):
        solution.append(value(slack_variable[i]))

    return solution


# modified for patition based learning
def call_ilp_solver(A, y, m, no_features, no_samples, weight_feature, weight_regularization, solution, return_dict):
    # todo (done) slack variable range changed, in constraints, eta is multiplied with slack variable to remain consistency
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)

    prob = LpProblem("m_of_n", LpMinimize)

    # declare variables & objective
    objective = []
    feature_variable = []
    if (len(solution) == 0):

        for i in range(no_features):
            feature_variable.append(LpVariable("a_" + str(i + 1), 0, 1, LpInteger))
            objective.append(weight_feature * feature_variable[i])
    else:
        for i in range(no_features):
            feature_variable.append(LpVariable("a_" + str(i + 1), 0, 1, LpInteger))
            if (solution[i] > 0):
                # in previous rule
                objective.append(-weight_feature * feature_variable[i])
            else:
                objective.append(weight_feature * feature_variable[i])
    slack_variable = []
    for i in range(no_samples):
        slack_variable.append(LpVariable("s_" + str(i + 1), 0, 1, LpInteger))
        objective.append(weight_regularization * slack_variable[i])

    # Objective
    prob += lpSum(objective)

    # Constraints

    for i in range(len(y)):
        constraint = []
        if (y[i] == 1):
            for j in range(len(A[i])):
                constraint.append(A[i][j] * feature_variable[j])

            constraint.append(no_features * slack_variable[i])

            prob += lpSum(constraint) >= m
        else:

            for j in range(len(A[i])):
                constraint.append(A[i][j] * feature_variable[j])

            constraint.append(-1 * no_features * slack_variable[i])

            prob += lpSum(constraint) <= m - 1
            slack_variable[i].upBound = None

    # call tentative solver

    # GLPK().solve(prob)
    prob.solve(CPLEX(msg=0))

    # parse solution solution
    solution = []
    for i in range(len(feature_variable)):
        solution.append(value(feature_variable[i]))

    for i in range(len(slack_variable)):
        solution.append(value(slack_variable[i]))

    return solution


if __name__ == "__main__":
    if (True):
        runTool()
    else:
        n = 3  # number of features
        m = 4  # number of samples
        eta = 1  # initially constant, later best choice using LP

        A = [
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]

        ]
        # print(A)

        y = [
            0,
            1,
            1,
            1
        ]
        # func()
        # call_LP_solver(A, y, eta, n, [1])
        call_ilp_solver(A, y, eta, n, len(y), [1])

        call_cp_solver(A, y, eta, n, len(y), [1])
        #
