from __future__ import print_function
from scipy.optimize import linprog
import time
import argparse
import pickle
import multiprocessing
from pulp import *
import numpy as np
import cplex

# from additional_scripts.cp_solver import call_generalized_cp_solver, call_multilevel_cp_solver

verbose = not True
learn_eta = False
memlimit = 1000*16


def call_cplex(A, y, eta, no_features, no_samples, weight_feature,
               weight_regularization, level,
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
            feature_variable.append(
                "b_" + str(i + 1) + str("_") + str(eachLevel + 1))

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
            variable_list.append("ax_" + str(i + 1) +
                                 str("_") + str(eachLevel + 1))

    # print(variable_list)
    # set variables
    myProblem.variables.add(names=variable_list)

    for eachLevel in range(level):
        for i in range(no_features):
            objective_coefficient.append(weight_feature)
            myProblem.variables.set_lower_bounds(variable_count, 0)
            myProblem.variables.set_upper_bounds(variable_count, 1)
            myProblem.variables.set_types(
                variable_count, myProblem.variables.type.integer)
            myProblem.objective.set_linear(
                [(variable_count, objective_coefficient[variable_count])])
            variable_count += 1

    # slack_variable = []
    for i in range(no_samples):
        objective_coefficient.append(weight_regularization)
        myProblem.variables.set_types(
            variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, 1)
        myProblem.objective.set_linear(
            [(variable_count, objective_coefficient[variable_count])])
        variable_count += 1

    myProblem.objective.set_sense(myProblem.objective.sense.minimize)

    var_eta_clause = -1

    if (learn_eta_clause):
        myProblem.variables.set_types(
            variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, level)
        var_eta_clause = variable_count
        variable_count += 1

    var_eta_literal = -1
    constraint_count = 0

    if (learn_eta):

        myProblem.variables.set_types(
            variable_count, myProblem.variables.type.integer)
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

                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.integer)
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
                            ind=[eachLevel * no_features +
                                 j for j in range(no_features)] + [variable_count],
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
                        ind=[i + level * no_features,
                             var_eta_clause] + auxiliary_index,
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
                        # 1st slack variable = level * no_features
                        ind=[i + level * no_features] + auxiliary_index,
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
                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.integer)
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
                            ind=[eachLevel * no_features +
                                 j for j in range(no_features)] + [variable_count],
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
                        ind=[i + level * no_features,
                             var_eta_clause] + auxiliary_index,
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
                        # 1st slack variable = level * no_features
                        ind=[i + level * no_features] + auxiliary_index,
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
    myProblem.parameters.workdir.set(
        "../Benchmarks/tempfiles_" + str(node) + "/")
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


def call_generalized_ilp_solver_clause_relaxation(A, y, eta, no_features, no_samples, weight_feature,
                                                  weight_regularization, level,
                                                  solver,
                                                  eta_clause, learn_eta_clause):
    # todo (done) slack variable range changed, in constraints, eta is multiplied with slack variable to remain consistency
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)
    # print(y)

    prob = LpProblem("m_of_n", LpMinimize)

    # declare variables & objective
    objective = []
    feature_variable = []
    for eachLevel in range(level):
        for i in range(no_features):
            feature_variable.append(LpVariable(
                "b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
            objective.append(weight_feature *
                             feature_variable[eachLevel * no_features + i])
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
                constraint.append(
                    feature_variable[eachLevel * no_features + j])

            prob += lpSum(constraint) >= var_eta_literal

    for i in range(len(y)):
        if (y[i] == 1):

            auxiliary_constraint = []

            for eachLevel in range(level):
                constraint = []
                for j in range(len(A[i])):
                    constraint.append(
                        A[i][j] * feature_variable[eachLevel * no_features + j])

                auxiliary_variables.append(
                    LpVariable("ax_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))

                constraint.append(no_features * auxiliary_variables[-1])
                auxiliary_constraint.append(auxiliary_variables[-1])

                if (learn_eta):
                    prob += lpSum(constraint) >= var_eta_literal
                else:
                    prob += lpSum(constraint) >= eta
            if (learn_eta_clause):
                prob += level * slack_variable[i] + level - \
                    var_eta_clause >= lpSum(auxiliary_constraint)
            else:
                prob += level * slack_variable[i] + level - \
                    eta_clause >= lpSum(auxiliary_constraint)
        else:

            auxiliary_constraint = []
            for eachLevel in range(level):

                constraint = []

                for j in range(len(A[i])):
                    constraint.append(
                        A[i][j] * feature_variable[eachLevel * no_features + j])

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
                prob += level * \
                    slack_variable[i] + \
                    var_eta_clause >= lpSum(auxiliary_constraint) + 1
            else:
                prob += level * \
                    slack_variable[i] + \
                    eta_clause >= lpSum(auxiliary_constraint) + 1
            # slack_variable[i].upBound = None
    # print(prob)
    # call tentative solver

    # print(len(auxiliary_variables))
    # select a solver
    if (solver == "default"):
        prob.solve()
    elif (solver == "gurobi"):
        prob.solve(GUROBI(msg=0))
    elif (solver == "cplex"):
        prob.solve(CPLEX())
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


def runTool():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset")
    parser.add_argument("--timeout", type=int,
                        help="cpu timeout in seconds", default=2000)
    parser.add_argument("--runIndex", type=int, help="runIndex", default=0)
    parser.add_argument("--m", type=int, help="m", default=2)
    parser.add_argument("--level", type=int, help="level", default=1)
    parser.add_argument(
        "--c", type=int, help="weight of feature vector", default=1)
    parser.add_argument("--lamda", type=int,
                        help="weight of regularization param", default=10)
    parser.add_argument("--node", type=int, help="node", default=0)
    parser.add_argument("--solver", type=str, help="solver", default="cplex")

    parser.add_argument("--eta_clause", type=int,
                        help="eta_clause", default=-1)

    # parser.add_argument("--classifier", type=str, help="Classifier name")

    args = parser.parse_args()
    level = args.level
    dataset = args.dataset
    timeOut = args.timeout - 500
    runIndex = args.runIndex
    weight_feature = args.c
    weight_regularization = args.lamda
    eta_clause = args.eta_clause

    # classifier = args.classifier
    eta = args.m

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
    # for runIndex in range(runIndex):

    data = pickle.load(open("../Benchmarks/tempfiles/" +
                            dataset+"_"+str(runIndex)+".dat", "rb"))
    X_train = data['Xtrain']
    y_train = data['ytrain']
    X_test = data['Xtest']
    y_test = data['ytest']
    column_names = data['columns']

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # column_names = [str(a) + str(" ") + str(b) + str(" ") + str(c) for (a, b, c) in
    #                 X_df.columns]

    start_time = time.time()

    if (args.solver != "cplex"):
        solution, eta, eta_clause = call_generalized_ilp_solver_clause_relaxation(X_train, list(y_train), eta,
                                                                                  len(X_train[0]), len(
                                                                                      y_train),
                                                                                  weight_feature,
                                                                                  weight_regularization, level,
                                                                                  args.solver,
                                                                                  eta_clause, learn_eta_clause)
    else:
        solution, eta, eta_clause = call_cplex(X_train, list(y_train), eta,
                                               len(X_train[0]), len(y_train),
                                               weight_feature,
                                               weight_regularization, level, args.solver,
                                               eta_clause, learn_eta_clause, timeOut, args.node)

    actual_complete += 1

    acc_train, compound_rule, rule_size = retrive_rule_and_train_acc(solution, len(X_train[0]), eta, column_names,
                                                                     level, eta_clause)

    # if (verbose):
    #     debug(level, X_train, y_train, solution)

    end_time = time.time()
    train_time = end_time - start_time
    start_time = time.time()
    acc_test, solution = find_test_acc(
        solution, X_test, y_test, eta, level, eta_clause)

    if (verbose):
        print("test\n\n\n")
        # debug(level, X_test, y_test, solution)

    end_time = time.time()
    test_time = end_time - start_time

    total_time_needed_train += train_time
    overall_acc_train += acc_train
    total_time_needed_test += test_time
    overall_acc_test += acc_test

    if (actual_complete == 0):
        total_time_needed_train = 2000
        overall_acc_train = 0
        total_time_needed_test = 2000
        overall_acc_test = 0

        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + str("Exact_LP") + " " \
              + str(dataset) + " " \
              + str(level) + " " \
              + str(weight_regularization) + " " \
              + str(args.solver) + " " \
              + str("NAN") + " " \
              + str(1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(-1) + " " \
              + str(runIndex) + " " \
              + str(eta) + " " + str("or") + " " + str(-1) + " NAN"

    else:
        total_time_needed_train = float(
            total_time_needed_train / actual_complete)
        overall_acc_train = float(overall_acc_train / actual_complete)
        total_time_needed_test = float(
            total_time_needed_test / actual_complete)
        overall_acc_test = float(overall_acc_test / actual_complete)

    # save result in files
    cmd = "python ../output/dump_result.py ../output/result.csv " \
          + str("Exact_LP") + " " \
          + str(dataset) + " " \
          + str(level) + " " \
          + str(weight_regularization) + " " \
          + str(args.solver) + " " \
          + str(rule_size) + " " \
          + str(1) + " " \
          + str(total_time_needed_train) + " " \
          + str(overall_acc_train * 100.0) + " " \
          + str(total_time_needed_test) + " " \
          + str(overall_acc_test * 100.0) + " " \
          + str(runIndex) + " " \
          + str(eta) + " " + str("or") + " " + \
        str(eta_clause) + "  " + str(compound_rule)

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
        print("rule size:                " + str(rule_size))
        print("train time:               " + str(total_time_needed_train))
        print("test time:                " + str(total_time_needed_test))
        print("train accuracy:           " + str(overall_acc_train * 100.0))
        print("test accuracy:            " + str(overall_acc_test * 100.0))
        # print("test accuracy:            " + str((1.0 - (float(hold_error) / float(total_batch_size_hold))) * 100.0))

        print("\nrule->")
        print(compound_rule)
        print("\n")

    # os.system("rm -R  ../Benchmarks/tempfiles_" + str(args.node))


def neg_feature(list, eta_optimization=False):
    neg_list = []
    for elem in list:
        neg_list.append(-elem)
    if (eta_optimization):
        neg_list.append(1)  # for eta
    return neg_list


def pos_feature(list, eta_optimization=False):
    if (eta_optimization):
        list.append(-1)  # for eta
    return list


def call_LP_solver(A, y, eta, n, return_dict, eta_optimization=False):
    '''

    :param A: {0,1} matrix
    :param y: label
    :return:
    '''

    print("no of features: ", n)
    print("no of samples : ", str(len(y)))
    print(y)
    A_ub = []
    b_ub = []
    c = [1 for i in range(n)]

    # adding slack variables
    reg_param = 1
    slack_variables = []
    bound_slack_variable = []
    for i in range(len(y)):
        c.append(reg_param)
        slack_variables.append(0)

    if (eta_optimization):
        c.append(1)

    # initial constraint

    all_variables = [-1 for i in range(n)] + slack_variables
    A_ub.append(all_variables)
    b_ub.append(-eta)
    print(all_variables)

    for i in range(len(y)):
        if (y[i] == 1):
            slack_variables[i] = 1
            all_variables = A[i] + slack_variables
            slack_variables[i] = 0
            A_ub.append(neg_feature(all_variables, eta_optimization))
            bound_slack_variable.append((0, eta))
            if (eta_optimization):
                b_ub.append(0)
            else:
                b_ub.append(-eta)
        else:
            slack_variables[i] = -1
            all_variables = A[i] + slack_variables
            slack_variables[i] = 0
            A_ub.append(pos_feature(all_variables, eta_optimization))
            bound_slack_variable.append((0, eta))

            if (eta_optimization):
                b_ub.append(-1)
            else:
                b_ub.append(eta - 1)

    bounds = []
    for i in range(n):
        bounds.append((0, 1))
    if (eta_optimization):
        bounds.append((1, n))
    # bound for slack variables
    bounds = bounds + bound_slack_variable

    # print(c)
    # print(A_ub)
    # print(b_ub)
    print(bounds)
    # for i in range(len(y)):
    #     print("hm")
    #     print(b_ub[i])
    #     print(y[i])
    #     print(A_ub[i])
    #     print(A[i])

    # res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
    #               options={"disp": True})
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex',
                  options={"disp": True, "maxiter": 5000})

    print(res.message)
    if (res.success == True):
        retrive_rule_and_train_acc(list(res.x), n, eta)
    return 0, 0, 0, 0
    # if (verbose):
    #     print(endTime_train - startTime_train, train_acc, endTime_test - startTime_test, test_acc)


def retrive_rule_and_train_acc(solution, no_features, eta, column_names, level, eta_clause):
    # print(x)
    # if (verbose):
    #     for variable in solution[:level * no_features]:
    #         if (variable != 0):
    #             print(variable)
    rule_size = 0
    rule = '[ ( '
    for eachLevel in range(level):

        for literal_index in range(no_features):
            if (solution[eachLevel * no_features + literal_index] >= 1):
                # rule += "  X_" + str(literal_index + 1) + "  +"
                rule += " " + column_names[literal_index] + "  +"
                rule_size += 1
        rule = rule[:-1]
        rule += ' )>= ' + str(eta) + "  ]"

        if (eachLevel < level - 1):
            rule += ' +\n[ ( '
    rule += "  >= " + str(eta_clause)
    # convert text for passing rule as an argument
    rule = rule.replace(" ", "^^")
    rule = rule.replace("(", "left_paren")
    rule = rule.replace(")", "right_paren")
    rule = rule.replace("<", "less_")
    rule = rule.replace(">", "greater_")
    rule = rule.replace("\n", "line_")

    # calculate error
    error_count = 0
    for index in range(no_features, len(solution)):
        if (solution[index] > 0):
            error_count += 1

    return float(len(solution) - no_features - error_count) / (len(solution) - no_features), rule, rule_size


def call_ilp_solver(A, y, eta, no_features, no_samples, weight_feature, weight_regularization, return_dict):
    # todo (done) slack variable range changed, in constraints, eta is multiplied with slack variable to remain consistency
    if (verbose):
        print("no of features: ", no_features)
        print("no of samples : ", no_samples)
    # print(y)

    prob = LpProblem("m_of_n", LpMinimize)

    # declare variables & objective
    objective = []
    feature_variable = []
    for i in range(no_features):
        feature_variable.append(LpVariable("a_" + str(i + 1), 0, 1, LpInteger))
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

            constraint.append(eta * slack_variable[i])

            prob += lpSum(constraint) >= eta
        else:

            for j in range(len(A[i])):
                constraint.append(A[i][j] * feature_variable[j])

            constraint.append(-1 * eta * slack_variable[i])

            prob += lpSum(constraint) <= eta - 1
            slack_variable[i].upBound = None

    # call tentative solver
    # print(prob)
    # GLPK().solve(prob)
    prob.solve(CPLEX(msg=0))
    # print(feature_variable)
    # print(prob)
    # parse solution solution
    solution = []
    for i in range(len(feature_variable)):
        solution.append(value(feature_variable[i]))

    for i in range(len(slack_variable)):
        solution.append(value(slack_variable[i]))

    return solution


def ParseFiles(datafile):
    file = open(datafile, "rb")
    x = pickle.load(file)
    AMatrix = x['A']
    yVector = x['y']
    A_df = x['A_df']
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
    file.close()
    return A_df, AMatrix, yVector, groupList, groupMap, len(AMatrix[0])


def find_test_acc(solution, A, y, eta, level, eta_clause):
    # matrix multiplication for calculating accuracy
    acc = 0
    # result = np.matmul(A, solution[:len(A[0])])
    # for i in range(len(y)):
    #     if (y[i] == 1 and result[i] >= eta):
    #         acc += 1
    #     if (y[i] == 0 and result[i] < eta):
    #         acc += 1
    # print(acc)
    acc = 0
    for i in range(len(y)):
        dot_value = [0 for eachLevel in range(level)]
        for j in range(len(A[i])):
            for eachLevel in range(level):
                dot_value[eachLevel] += A[i][j] * \
                    solution[eachLevel * len(A[i]) + j]
        if (y[i] == 1):
            correctClauseCount = 0
            for eachLevel in range(level):
                if (dot_value[eachLevel] >= eta):
                    correctClauseCount += 1
            if (correctClauseCount >= eta_clause):
                acc += 1
                # solution[len(A[i]) * level + i] = 0
            # else:
            #     solution[len(A[i]) * level + i] = 1
        else:
            correctClauseCount = 0
            for eachLevel in range(level):
                if (dot_value[eachLevel] < eta):
                    correctClauseCount += 1
            if (correctClauseCount > level - eta_clause):
                acc += 1
                # solution[len(A[i]) * level + i] = 0
            # else:
            #     solution[len(A[i]) * level + i] = 1
        # solution = solution[:len(A[i]) * level + len(y)]

        # if (dot_value >= eta and y[i] == 1):
        #     acc += 1
        # elif (dot_value < eta and y[i] == 0):
        #     acc += 1

    # print(acc)
    return float(acc) / len(y), solution


def debug(level, X, y, solution):
    xSize = len(X[0])
    for each_level in range(level):
        print(each_level)
        for i in range(len(y)):
            for j in range(xSize):
                if (solution[j + each_level * xSize] == 1):
                    print(X[i][j], end=' '),
            print(y[i], end=' '),
            print(int(solution[i + xSize * level]))


def call_generalized_ilp_solver(A, y, eta, no_features, no_samples, weight_feature, weight_regularization, level,
                                solver,
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
    for eachLevel in range(level):
        for i in range(no_features):
            feature_variable.append(LpVariable(
                "b_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
            objective.append(weight_feature *
                             feature_variable[eachLevel * no_features + i])
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
                    constraint.append(
                        A[i][j] * feature_variable[eachLevel * no_features + j])

                constraint.append(no_features * slack_variable[i])

                prob += lpSum(constraint) >= eta
        else:

            auxiliary_constraint = []
            for eachLevel in range(level):

                constraint = []

                for j in range(len(A[i])):
                    constraint.append(
                        A[i][j] * feature_variable[eachLevel * no_features + j])

                auxiliary_variables.append(
                    LpVariable("ax_" + str(i + 1) + str("_") + str(eachLevel + 1), 0, 1, LpInteger))
                # attach the last aux variable
                constraint.append(-1 * no_features * auxiliary_variables[-1])
                auxiliary_constraint.append(auxiliary_variables[-1])
                prob += lpSum(constraint) <= eta - 1

            # constraint for aux variables
            prob += level * \
                (1 + slack_variable[i]) >= lpSum(auxiliary_constraint) + 1
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


if __name__ == "__main__":
    if (True):
        runTool()
    