# find test accuracy, train accuracy, time, rule size
# for the row which has best avg test acc
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

markersize = 20
fontsize = 32
labelsize = 20


def plot_acc_vs_rulesize(acc, rulesize, dataset):

    plt.title(dataset, fontsize=fontsize)
    plt.ylabel("accuracy", fontsize=fontsize)
    plt.xlabel("rule size", fontsize=fontsize)
    plt.scatter(rulesize, acc)
    plt.tight_layout()
    plt.tick_params(labelsize=labelsize)
    # plt.show()
    plt.savefig("acc_vs_rulesize_"+dataset+".pdf")
    plt.clf()


def print_rule():

    # print rule that has test accuracy at least equal to best median test accuracy
    df_2 = pd.read_csv("merge.csv", header=0)

    # find best rule    
    df = pd.read_csv("cross_validation.csv", header=0)
    df.columns = ['dataset', 'method', 'level', 'lambda', 'partition', 'threshold',
                  'train accuracy', 'test accuracy', 'time', 'rule size', 'mad test accuracy', 'mad rule size', 'appearance',
                  ' mean test accuracy', 'mean train accuracy', 'mean time', 'mean rule size', 'sem test accuracy', 'sem rule size']


    select_type = "test accuracy"
    group_list = ["method", "dataset"]
    idx = df.groupby(group_list)[
        select_type].transform(max) == df[select_type]
    result = df[idx]
    result = pd.DataFrame(result[df.columns], copy=True)
    # consider row that has minimum rule size

    select_type = "rule size"
    group_list = ["method", "dataset"]
    idx = result.groupby(group_list)[select_type].transform(
        min) == result[select_type]
    result = result[idx]
    result = pd.DataFrame(result[result.columns], copy=True)

    group_list = ["dataset",  "method", "level",
                  "lambda",   "partition",  "threshold"]



    common = df_2.merge(result, on=group_list)
    
    # common[(common['method']=='Approx_LP') & (common['dataset']=='heart')].to_csv("heart.csv")
    
    grouped_df = common.groupby(["method", "dataset"])
    for key, item in grouped_df:

        method, dataset = key
        if(method=="Approx_LP"):
            method="inc-IRR"
        if(method=="Approx"):
            method="IMLI"
        if(method=="Exact_LP"):
            method="IRR"
        if(method!="IRR" and method!="inc-IRR" and method!="IMLI"):
            continue
        print("\n\n", method, ":", dataset, "\n\n")
        rules = grouped_df.get_group(key)['rule'].unique()
        rule_freq = grouped_df.get_group(key)['rule'].value_counts().to_dict()
        cnt = 1
        for rule, freq in rule_freq.items():
            rule = rule.replace("line_", "\n")
            rule = rule.replace("greater_", ">")
            rule = rule.replace("less_", "<")
            rule = rule.replace("right_paren", ")")
            rule = rule.replace("left_paren", "(")
            rule = rule.replace("^^", " ")
            rule = rule.replace("_ge_", ">=").replace(
                "_l_", "<").replace("_eq_", "=")
            print(str(cnt),  rule, "\n\n")

            cnt += 1


def compute_from_csv():
    df = pd.read_csv("time.csv", header=None)
    df.columns = ["method", "dataset", "level", "lambda", "solver", "rule size", "partition", "runIndex", "threshold",
                            "time"]
    df.drop(["solver", "rule size"], axis=1, inplace=True)

    df.dropna(how='any', axis=0, inplace=True)

    # delete smaller datasets
    df = df[df["dataset"] != "iris"]
    df = df[df["dataset"] != "transfusion"]

    df['time'] = pd.to_numeric(df['time'])
    df.reset_index(drop=True, inplace=True)

    df_2 = pd.read_csv("result.csv", header=None)
    df_2.columns = ["method", "dataset", "level", "lambda", "solver", "rule size", "partition", "train time",
                    "train accuracy", "test time", "test accuracy", "runIndex", "threshold", 'rule_type', 'eta_clause', 'rule', 'NAN']
    df_2.drop(["solver", 'NAN', 'rule_type'], axis=1, inplace=True)
    df.dropna(how='any', axis=0, inplace=True)

    df_2.reset_index(drop=True, inplace=True)

    # partition=NAN to 1
    df['partition'].replace(['NAN'], [1], inplace=True)
    df['partition'] = pd.to_numeric(df['partition'])
    df_2['partition'].replace(['NAN'], [1], inplace=True)
    df_2['partition'] = pd.to_numeric(df_2['partition'])

    # for Approx, Approx_LP, Exact_LP, ripper if rule size=0, remove those rows
    df_2 = df_2[df_2['rule size'] != "0"]

    # join two dataframes
    common = df.merge(df_2, on=["method", "dataset", "level",
                                "lambda",   "partition", "runIndex", "threshold"])

    # change rule size = NAN to 0
    common['rule size'].replace(['NAN'], [0], inplace=True)
    common['rule size'] = pd.to_numeric(common['rule size'])

    # common.to_csv("common.csv")

    common.dropna(how='any', axis=0, inplace=True)
    # print(df.shape)
    # print(df_2.shape)

    # # perform groupby on other classifier

    group_list = ["dataset",  "method", "level",
                  "lambda",   "partition",  "threshold"]

    mad_test = common.groupby(group_list)["test accuracy"].mad()
    mad_rule_size = common.groupby(group_list)["rule size"].mad()

    median_time = common.groupby(group_list)["time"].median()
    median_train_time = common.groupby(group_list)["train time"].median()
    median_train_accuracy = common.groupby(
        group_list)["train accuracy"].median()
    median_test_time = common.groupby(group_list)["test time"].median()
    median_test_accuracy = common.groupby(group_list)["test accuracy"].median()
    median_rule_size = common.groupby(group_list)["rule size"].median()
    max_test_accuracy = common.groupby(group_list)["test accuracy"].max()
    max_rule_size = common.groupby(group_list)["rule size"].max()
    min_rule_size = common.groupby(group_list)["rule size"].min()
    appearance = common.groupby(group_list)["time"].count()
    mean_test_accuracy = common.groupby(group_list)["test accuracy"].mean()
    mean_train_accuracy = common.groupby(group_list)["train accuracy"].mean()
    mean_time = common.groupby(group_list)["time"].mean()
    sem_test_accuracy = common.groupby(group_list)["test accuracy"].sem()
    mean_rule_size = common.groupby(group_list)["rule size"].mean()
    sem_rule_size = common.groupby(group_list)["rule size"].sem()

    common.to_csv("merge.csv")

    result_df = pd.concat(
        [median_train_accuracy, median_test_accuracy, median_time, median_rule_size, mad_test, mad_rule_size, appearance, mean_test_accuracy, mean_train_accuracy,
         mean_time, mean_rule_size, sem_test_accuracy, sem_rule_size], axis=1)
    # result_df.columns=['median accuracy','max accuracy',"time","rule size median","rule size max","rule size min"]
    # result_df['rule size'].replace([0],['-'], inplace=True)
    # print(result_df)
    result_df = result_df.round(2)
    # print(result_df)
    result_df.to_csv("cross_validation.csv")


def find_best_result():

    # already median accuracy is calculated, this is the estimate accuracy
    # now find parameter choice that has max accuracy and report that accuracy and rule size
    # if there is a tie while calculating max, take the one with minimum rule size

    df = pd.read_csv("cross_validation.csv", header=0)
    df.columns = ['dataset', 'method', 'level', 'lambda', 'partition', 'threshold',
                  'train accuracy', 'test accuracy', 'time', 'rule size', 'mad test accuracy', 'mad rule size', 'appearance',
                  ' mean test accuracy', 'mean train accuracy', 'mean time', 'mean rule size', 'sem test accuracy', 'sem rule size']

    df[(df["method"] == "Approx_LP") & (
        df["dataset"] == "heart")].to_csv("heart.csv")

    # plot accuracy vs rule size
    # datasets=df['dataset'].unique()
    # for dataset in datasets:
    #     plot_df=df[(df['method']=="Approx_LP") & (df['dataset']==dataset)]
    #     plot_acc_vs_rulesize(list(plot_df['test accuracy']),list(plot_df['rule size']),dataset)

    # all rows that have max test accuracy

    select_type = "test accuracy"
    group_list = ["method", "dataset"]
    idx = df.groupby(group_list)[
        select_type].transform(max) == df[select_type]
    result = df[idx]
    result = pd.DataFrame(result[df.columns], copy=True)

    # consider row that has minimum rule size

    select_type = "rule size"
    group_list = ["method", "dataset"]
    idx = result.groupby(group_list)[select_type].transform(
        min) == result[select_type]
    result = result[idx]
    result = pd.DataFrame(result[result.columns], copy=True)

    # take median of all now

    median_time = result.groupby(group_list)["time"].median()
    median_train_accuracy = result.groupby(
        group_list)["train accuracy"].median()
    median_rule_size = result.groupby(group_list)["rule size"].median()
    median_mad_test_accuracy = result.groupby(
        group_list)["test accuracy"].mad()
    max_test_accuracy = result.groupby(group_list)["test accuracy"].median()

    result = pd.concat(
        [max_test_accuracy,  median_time, median_rule_size, median_train_accuracy], axis=1)

    result = result.round(2)

    result.to_csv("best_result.csv")


def parameter_vary(param):

    # already median accuracy is calculated, this is the estimate accuracy
    # now find parameter choice that has max accuracy and report that accuracy and rule size
    # if there is a tie while calculating max, take the one with minimum rule size

    # find the best parameter choice, now vary param, take mean and standard error

    df = pd.read_csv("cross_validation.csv", header=0)
    df.columns = ['dataset', 'method', 'level', 'lambda', 'partition', 'threshold',
                  'train accuracy', 'test accuracy', 'time', 'rule size', 'mad test accuracy', 'mad rule size', 'appearance',
                  'mean test accuracy', 'mean train accuracy', 'mean time', 'mean rule size', 'sem test accuracy', 'sem rule size']

    # all rows that have max test accuracy

    select_type = "test accuracy"
    group_list = ["method", "dataset"]
    idx = df.groupby(group_list)[
        select_type].transform(max) == df[select_type]
    result = df[idx]
    result = pd.DataFrame(result[df.columns], copy=True)

    # consider row that has minimum rule size

    select_type = "rule size"
    group_list = ["method", "dataset"]
    idx = result.groupby(group_list)[select_type].transform(
        min) == result[select_type]
    result = result[idx]
    result = pd.DataFrame(result[result.columns], copy=True)

    # take intersection of best result and given result

    merge_params = list(set(["method", "dataset", "level",
                             "lambda",   "partition",  "threshold"])-set([param]))

    result = result[merge_params]
    common = df.merge(result, on=merge_params)

    # common.to_csv("parameter_effect.csv")

    # select methods for showing results
    common = common[(common['method'] == "Exact_LP") |
                    (common['method'] == "Approx_LP")]

    group_list = ["method", 'dataset']
    pd.options.mode.chained_assignment = None
    grouped_df = common.groupby(group_list)
    for key, item in grouped_df:
        # print("\n\n", key, "\n\n")
        method, dataset = key
        item[param] = item[param].apply(pd.to_numeric)

        item = item.sort_values([param], ascending=True)
        if(param == "partition" and method == "Exact_LP"):
            selected_df = df[df['method'] == "Approx_LP"].merge(item[["dataset", "level",
                                                                      "lambda",    "threshold"]], on=["dataset", "level",
                                                                                                      "lambda",    "threshold"])
            # print(selected_df)
            # print(item)
            # print("\n\n\n")

            selected_df[[param, "mean test accuracy", "mean train accuracy",
                         "mean time", "mean rule size", "sem test accuracy", "sem rule size"]].to_csv(dataset+"_"+method+"_params.csv", mode='a')
            item = item[[param, "mean test accuracy", "mean train accuracy",
                         "mean time", "mean rule size", "sem test accuracy", "sem rule size"]]
            item.to_csv(dataset+"_"+method+"_params.csv", mode='a')
            # print(selected_df.iloc[[0]][[param, "mean test accuracy", "mean train accuracy",
            #                              "mean time", "mean rule size", "sem test accuracy", "sem rule size"]])
        else:

            item = item[[param, "mean test accuracy", "mean train accuracy",
                         "mean time", "mean rule size", "sem test accuracy", "sem rule size"]]
            item.to_csv(dataset+"_"+method+"_params.csv", mode='a')


def show_result():

    df = pd.read_csv("best_result.csv", header=0)
    print(df)
    df.columns = ["method", "dataset", "test accuracy",
                  "time", "rule size", "train accuracy"]

    df = df.groupby(["dataset"])

    datasets = []
    size = []
    features = []

    test_acc = {}
    train_acc = {}
    time = {}
    rule_size = {}

    # print(df.transpose())

    for key, item in df:
        # print(key)
        # print(item)

        if(key == 'adult'):
            size.append(32561)
            features.append(144)
            datasets.append("Adult")
        elif(key == 'credit'):
            size.append(30000)
            features.append(110)
            datasets.append("Credit")
        elif(key == 'ionosphere'):
            size.append(351)
            features.append(144)
            datasets.append("Ionosphere")
        # if(key == 'iris'):
        #     size.append(150)
        #     features.append(11)
        #     datasets.append("Iris")
        elif(key == 'parkinsons'):
            size.append(195)
            features.append(51)
            datasets.append("Parkinsons")
        elif(key == 'pima'):
            size.append(768)
            features.append(30)
            datasets.append("Pima")
        elif(key == 'toms'):

            size.append(28179)
            features.append(910)
            datasets.append("Tom\'s HW")
        if(key == 'transfusion'):
            size.append(748)
            features.append(6)
            datasets.append("Blood")
        elif(key == 'twitter'):
            size.append(49999)
            features.append(1511)
            datasets.append("Twitter")
        elif(key == 'wdbc'):
            size.append(569)
            features.append(88)
            datasets.append("WDBC")
        elif(key == 'tictactoe'):
            size.append(958)
            features.append(27)
            datasets.append("Tic Tac Toe")
        elif(key == 'compas'):
            size.append(7210)
            features.append(19)
            datasets.append("Compas")
        elif(key == 'titanic'):
            size.append(1309)
            features.append(26)
            datasets.append("Titanic")
        if(key == 'iris'):
            size.append(150)
            features.append(11)
            datasets.append("Iris")
        elif(key == 'heart'):
            size.append(303)
            features.append(31)
            datasets.append("Heart")
        elif(key == 'ilpd'):
            size.append(583)
            features.append(14)
            datasets.append("ILPD")
        # print(key)

        mask = (item["method"] == "Exact_LP")
        if(True in mask.values.tolist()):
            classifier = "ours"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "Approx_LP")
        if(True in mask.values.tolist()):
            classifier = "inc_ours"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "Approx")
        if(True in mask.values.tolist()):
            classifier = "imli"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "logreg")
        if(True in mask.values.tolist()):
            classifier = "logreg"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "nn")
        if(True in mask.values.tolist()):
            classifier = "nn"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "ripper")
        if(True in mask.values.tolist()):
            classifier = "ripper"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "rf")
        if(True in mask.values.tolist()):
            classifier = "rf"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "svc")
        if(True in mask.values.tolist()):
            classifier = "svc"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]

        mask = (item["method"] == "BOA")
        if(True in mask.values.tolist()):
            classifier = "BOA"
            test_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                            "test accuracy"].values[0]
            rule_size[(classifier, datasets[-1])] = item.loc[mask,
                                                             "rule size"].values[0]
            train_acc[(classifier, datasets[-1])] = item.loc[mask,
                                                             "train accuracy"].values[0]
            time[(classifier, datasets[-1])
                 ] = item.loc[mask, "time"].values[0]
    sorted_dataset_index = [i[0]
                            for i in sorted(enumerate(size), key=lambda x:x[1])]

    print("\n\n\n", "Copy the following lines into a tex file\n\n")

    print("\\documentclass{article}\n\\usepackage{multirow}\n\\usepackage{amsmath}\n\\begin{document}")
    
    print("\\begin{table*}[h]		\n\\setlength{\\tabcolsep}{3pt}\n \\scriptsize \n	\\begin{center}")    

    print("\\begin{tabular}{|l | r | r |r |r| r |r |r |r| r|r|}\n\\hline	\n {{Dataset}} & Size & Features & LR & NN & SVC & RF & RIPPER  &  {IMLI}  & {IRR}& inc-{IRR} \\\\	\n\\hline")
    for count in sorted_dataset_index:

        print("\\multirow{2}{*}{", datasets[count], end="}  ")
        print(" & $ ", size[count], end=" $ ")
        print(" & $ ", features[count], end=" $ ")

        l_paren = ""
        r_paren = ""
        classifier = "logreg"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "nn"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "svc"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "rf"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "ripper"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="   ")
        # classifier = "BOA"
        # if((classifier, datasets[count]) in test_acc):
        #     print(" & $ ", l_paren, test_acc[(
        #         classifier, datasets[count])], r_paren, end=" $   ")
        # else:
        #     print(" & ", l_paren, "", r_paren, end="   ")

        classifier = "imli"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "ours"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        classifier = "inc_ours"
        if((classifier, datasets[count]) in test_acc):
            print(" & $ ", l_paren, test_acc[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print(" & ", l_paren, "", r_paren, end="    ")

        print("\\\\ & & ")

        classifier = "logreg"

        l_paren = "("
        r_paren = ")"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "nn"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "svc"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "rf"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "ripper"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        # classifier = "BOA"
        # if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
        #     print(" & $ ", l_paren,
        #           time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        # else:
        #     print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "imli"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "ours"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        classifier = "inc_ours"
        if((classifier, datasets[count]) in time and time[(classifier, datasets[count])] <= 2000):
            print(" & $ ", l_paren,
                  time[(classifier, datasets[count])], "\\text{s}", r_paren, end=" $   ")
        else:
            print(" & $ ", l_paren, " 2000 \\text{s} " ,r_paren, end=" $    ")

        print("\\\\ \\hline")

    
    print("\\end{tabular}\n\\end{center}\n\\caption{Comparisons of test accuracy and training time for ten fold cross-validation for different classifiers. Every cell in the last eight columns contain the best test accuracy ($ \\% $) and corresponding train time in seconds (within  parentheses).}\n\\end{table*}")
    
    print("\\begin{table*}[h]		\n	\\begin{center}")    

    print("\\begin{tabular}{|l |  r |r |r | }\n\\hline\n{Dataset}  & RIPPER  &  {IMLI}  &  inc-{IRR}\\\\\\hline		")

    for count in sorted_dataset_index:

        print("\\multirow{1}{*}{", datasets[count], end="}  ")
        l_paren = ""
        r_paren = ""

        classifier = "ripper"
        if((classifier, datasets[count]) in rule_size):
            print(" & $ ", l_paren, rule_size[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print("  &", l_paren, "", r_paren, end="    ")
        # classifier = "BOA"
        # if((classifier, datasets[count]) in rule_size):
        #     print(" & $ ", l_paren, rule_size[(
        #         classifier, datasets[count])], r_paren, end=" $   ")
        # else:
        #     print("  &", l_paren, "", r_paren, end="    ")

        classifier = "imli"
        if((classifier, datasets[count]) in rule_size):
            print(" & $ ", l_paren, rule_size[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print("  &", l_paren, "", r_paren, end="   ")

        # classifier="ours"
        # if((classifier,datasets[count]) in rule_size):
        # 	print(" & $ ",l_paren, rule_size[(classifier,datasets[count])], r_paren, end= " $   ")
        # else:
        # 	print("  ",l_paren, "TO", r_paren, end="  &  ")

        classifier = "inc_ours"
        if((classifier, datasets[count]) in rule_size):
            print(" & $ ", l_paren, rule_size[(
                classifier, datasets[count])], r_paren, end=" $   ")
        else:
            print("  &", l_paren, "", r_paren, end="   ")

        print("\\\\ \\hline")
    print("\\end{tabular}\n\\end{center}\n\\caption{Size of the rules generated by  interpretable classifiers.}	\n\\end{table*}")
    print("\\end{document}")

compute_from_csv()
find_best_result()
# params = ['lambda', "level", "partition", "threshold"]
# for param in params:
#     print(param)
#     parameter_vary(param)
print_rule()
show_result()
