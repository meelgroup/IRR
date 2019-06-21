import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np, scipy as sp, pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, metrics, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score as AUC, accuracy_score

import subprocess, os, re, time, argparse
import pickle
import multiprocessing


#############################################
def load_check_data(fname_tr, fname_tst):
    print("Loading file %s" % fname_tr)
    X_tr_pd = pd.read_table(fname_tr, sep=',', header=0)
    print("Loading file %s" % fname_tst)
    X_tst_pd = pd.read_table(fname_tst, sep=',', header=0)

    ### some data-sets (e.g. ilpd) have missing values (nans) -- fix these by replacing with 0's
    X_tr_pd.fillna(0, inplace=True)
    X_tst_pd.fillna(0, inplace=True)

    assert np.all(X_tr_pd.columns == X_tst_pd.columns), "Need X_tr and X_tst to have same features"

    N_tr = len(X_tr_pd)

    ### convert the format together --> then split
    X_tst_pd.index = X_tst_pd.index + N_tr
    X_pd_all = pd.concat([X_tr_pd, X_tst_pd])

    target_name = X_pd_all.columns[-1]

    # X_pd_all = X_pd_all.iloc[:100,:]
    X_pd_dict = X_pd_all.T.to_dict().values()

    vec = DictVectorizer()
    X_all = vec.fit_transform(X_pd_dict).toarray()
    feat_names = vec.get_feature_names()
    num_feat = len(feat_names)

    target_ind = feat_names.index(target_name)
    feat_names = feat_names.remove(target_name)

    y_tr = X_all[:N_tr, target_ind].flatten()
    y_tst = X_all[N_tr:, target_ind].flatten()

    ## Split back into train and test
    inds_feat = list(range(num_feat))
    inds_feat.remove(target_ind)
    inds_feat = np.array(inds_feat)

    X_tr = X_all[:N_tr, inds_feat]
    X_tst = X_all[N_tr:, inds_feat]

    return X_tr, X_tst, y_tr, y_tst, feat_names, target_name


#############################################
def classify_exp(X_tr, X_tst, y_tr, y_tst):
    ''' train and evaluate a few classifiers '''

    ### We need to tune the params...!! (setting at some detault values for now)
    min_samp = 10
    rf = RandomForestClassifier(n_estimators=500, min_samples_split=min_samp, n_jobs=-1)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': min_samp,
              'learning_rate': 0.01, 'loss': 'deviance'}
    gb = GradientBoostingClassifier(**params)

    C = 0.01
    logreg = linear_model.LogisticRegression(penalty='l1', C=C)
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C, probability=True)

    classifiers = {"rf": rf, "logreg": logreg}  ## "GradBoost": gb,"SVC" : svc
    results = {}  ### save results for all classifiers

    for cl_name in classifiers.keys():
        cl = classifiers[cl_name]

        cl.fit(X_tr, y_tr)
        y_hat = cl.predict(X_tst)
        p = cl.predict_proba(X_tst)

        acc = accuracy_score(y_tst, y_hat)
        auc = AUC(y_tst, p[:, 1])

        results[cl_name] = {'acc': acc, 'auc': auc}

        print("%s: accuracy: %.2f, AUC: %.2f" % (cl_name, acc, auc))

    return results


#############################################
def sample_data_classify_exp():
    ''' run a few classifiers on sample train and test data '''

    data_dir = os.getcwd() + "/"

    # exp_name = 'adult_data_bintarget'
    exp_name = 'ilpd';

    num_exp = 10
    all_results = []

    for enum in range(num_exp):
        fname_tr = data_dir + 'Train/' + '%s_%d_train.csv' % (exp_name, enum)
        fname_tst = data_dir + 'Test/' + '%s_%d_test.csv' % (exp_name, enum)

        X_tr, X_tst, y_tr, y_tst, feat_names, target_name = load_check_data(fname_tr, fname_tst)

        results = classify_exp(X_tr, X_tst, y_tr, y_tst)
        all_results.append(results)

    ### now summarize the results
    all_cl = all_results[0].keys()
    num_cl = len(all_cl)
    all_acc = np.zeros([num_exp, num_cl])
    all_auc = np.zeros([num_exp, num_cl])
    for cl_ind, cl in enumerate(all_cl):
        for exp in range(num_exp):
            all_acc[exp, cl_ind] = all_results[exp][cl]['acc']
            all_auc[exp, cl_ind] = all_results[exp][cl]['auc']

    print("Acc:")
    print(all_acc)
    print("AUC:")
    print(all_auc)
    print("")


#############################################
def gen_logreg(C_grid=[]):
    ''' generator of logreg classifiers param. by C'''

    if len(C_grid) == 0:
        C_grid = np.logspace(-3, 1, 10)

    for C in C_grid:
        logreg = linear_model.LogisticRegression(penalty='l1', C=C)

        yield (logreg, C)


#############################################
def gen_rf(min_samp_grid=[]):
    ''' generator of rf classifiers param. by min_leaf'''

    if len(min_samp_grid) == 0:
        min_samp_grid = [3, 5, 10, 25, 50, 100, 250, 500]

    for min_samp in min_samp_grid:
        rf = RandomForestClassifier(n_estimators=500, min_samples_split=min_samp, n_jobs=-1)

        yield (rf, min_samp)


#############################################
def gen_nn(num_nbs_grid=[], max_nn=None, dist_p=1):
    ''' generator of nearest-neighbors classifiers param. by num-nbs'''

    if len(num_nbs_grid) == 0:
        num_nbs_grid = [1, 3, 5, 11, 25, 51, 101, 501]

    if max_nn:
        num_nbs_grid = [x for x in num_nbs_grid if x <= max_nn]

    for nn in num_nbs_grid:
        knn = KNeighborsClassifier(n_neighbors=nn, p=dist_p)

        yield (knn, nn)


#############################################
def gen_svc(C_grid=[]):
    ''' generator of SVC classifiers param. by C'''

    if len(C_grid) == 0:
        C_grid = np.logspace(-3, 1, 10)

    for C in C_grid:
        svc = svm.SVC(kernel='linear', C=C, probability=True)

        yield (svc, C)


##################
def ParseFiles(datafile):
    file = open(datafile, "rb")
    x = pickle.load(file)
    AMatrix = x['A']
    yVector = x['y'].values.ravel()
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


#############################################
def run_all_classifiers_cv(cl_name, dataset, node, runIndex, param_index):
    ''' get results for all the classifiers '''

    data_dir = "../Benchmarks/tempfiles/"

    results = []
    timeTakenData = []

    data = pickle.load(open("../Benchmarks/tempfiles/" +
                        dataset+"_"+str(runIndex)+".dat", "rb"))
    X_tr = data['Xtrain']
    y_tr = data['ytrain']
    X_tst = data['Xtest']
    y_tst = data['ytest']
    column_names = data['columns']

    if cl_name == 'rf':
        gen_cl = gen_rf;
        cv_param = 'min_samp'
    elif cl_name == 'logreg':
        gen_cl = gen_logreg;
        cv_param = 'C'
    elif cl_name == 'nn':
        gen_cl = lambda: gen_nn(max_nn=len(y_tr));
        cv_param = 'num_nbs'
    elif cl_name == 'svc':
        gen_cl = gen_svc;
        cv_param = 'C'
    else:
        assert False, "No such classifier %s" % cl_name
    param_wait_index = 0

    for cl, min_samp in gen_cl():
        if (not (param_wait_index == param_index)):
            param_wait_index += 1
            continue
        param_wait_index += 1
        startTime = time.time()
        cl.fit(X_tr, y_tr)
        endTime = time.time()
        timeTaken = endTime - startTime
        y_hat = cl.predict(X_tst)
        p = cl.predict_proba(X_tst)

        acc = accuracy_score(y_tst, y_hat)
        auc = AUC(y_tst, p[:, 1])

        y_hat = cl.predict(X_tr)
        p = cl.predict_proba(X_tr)

        tr_acc = accuracy_score(y_tr, y_hat)
        tr_auc = AUC(y_tr, p[:, 1])

        results.append([runIndex, min_samp, acc, auc, tr_acc, tr_auc])
        timeTakenData.append([runIndex, min_samp, timeTaken])

    results_pd = pd.DataFrame(results, index=None, columns=['exp_num', cv_param, 'acc', 'auc', 'tr_acc', 'tr_auc'])
    timeTaken_pd = pd.DataFrame(timeTakenData, index=None, columns=['exp_num', cv_param, 'Time'])

    av_cv = results_pd.groupby(cv_param).mean()
    av_time = timeTaken_pd.groupby(cv_param).mean()
    del av_cv['exp_num']
    del av_time['exp_num']
    return  av_cv, av_time


############################################
def parse_weka_training_out(weka_out):
    weka_out = weka_out.decode("utf-8")

    andCount = 0
    shouldConsiderRuleCount = False
    learnedRules = ''
    for line in weka_out.split('\n'):
        line = line.strip()
        if re.match('Number of Rules', line):
            shouldConsiderRuleCount = False
        if (re.match('Correctly Classified Instances', line)):
            field_list = line.split()
            training_accuracy = float(field_list[-2]) / 100
            return andCount, training_accuracy, learnedRules
        if (shouldConsiderRuleCount):
            if (not (bool(re.search('=>', line)))):
                continue
            learnedRules += line + '\n'
            field_list = line.split(' and ')
            andCount += len(field_list)
        if (re.match('JRIP rules', line)):
            shouldConsiderRuleCount = True

    return 0


#############################################
def parse_weka_out(weka_out, do_debug=False):
    '''  parse the weird output weka file '''
    weka_out = weka_out.decode("utf-8")

    y_true, y_hat = [], []

    for line in weka_out.split('\n'):
        line = line.strip()
        if re.match('=== Predictions', line):
            continue
        if re.match('inst#', line):
            continue
        if len(line) == 0:
            continue

        field_list = line.split()
        if len(field_list) == 5:
            del field_list[3]

        if do_debug:
            print(' | '.join(field_list))

        y_true_ln = field_list[1].split(':')[1]
        y_hat_ln = field_list[2].split(':')[1]
        y_true.append(y_true_ln)
        y_hat.append(y_hat_ln)

    y_true_arr = np.array(y_true)
    y_hat_arr = np.array(y_hat)
    acc = np.sum(y_true_arr == y_hat_arr) * 1.0 / len(y_hat_arr)

    return acc


#############################################
def run_weka_RIPPER(cl_name, dataset, param_index, runIndex, node):
    ''' get results for weka RIPPER '''

    data_dir = os.getcwd() + '/../Benchmarks/tempfiles/'
    data_dir_temp = os.getcwd() + '/../Benchmarks/tempfiles_' + str(node) + "/"

    weka_cp = os.environ.get('WEKA_PATH') + 'weka.jar'
    weka_cl = 'weka.classifiers.rules.JRip'

    results = []
    timeTakenData = []
    ruleText = ''

    fname_tr = data_dir + str(dataset) + "_train_" + str(runIndex) + ".arff"
    fname_tst = data_dir + str(dataset) + "_test_" + str(runIndex) + ".arff"

    weka_model_fname = data_dir_temp + str(dataset) + "_" + str(runIndex) + "_out.model"

    # print(weka_model_fname)
    current_param = 0
    for min_leaf in [1, 2, 3, 5, 10, 15, 25, 50, 100]:
        current_param += 1
        if (param_index != current_param - 1):
            continue
        weka_tr_cmd = 'java -Xmx4G -cp %s %s -N %d -t %s -no-cv -c -1 -d %s' % (
            weka_cp, weka_cl, min_leaf, fname_tr, weka_model_fname)

        startTime = time.time()
        p = subprocess.Popen(weka_tr_cmd, stdout=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()
        endTime = time.time()
        timeTaken = endTime - startTime
        andCount, training_accuracy, learnedRules = parse_weka_training_out(out)

        weka_tst_cmd = 'java  -Xmx4G -cp %s %s -l %s -T %s -p 0' % (
            weka_cp, weka_cl, weka_model_fname, fname_tst)
        p = subprocess.Popen(weka_tst_cmd, stdout=subprocess.PIPE, shell=True)
        (weka_out, err) = p.communicate()

        acc = parse_weka_out(weka_out)

        ruleText += 'Exp_' + str(runIndex) + '_' + str(min_leaf) + ' ' + str(acc) + ' ' + str(andCount) + '\n'
        ruleText += learnedRules + '\n'
        results.append([runIndex, min_leaf, acc, andCount, training_accuracy])
        timeTakenData.append([runIndex, min_leaf, timeTaken])

    results_pd = pd.DataFrame(results, index=None, columns=['exp_num', 'min_leaf', 'acc', 'RuleSize', 'tr_acc'])
    timeTaken_pd = pd.DataFrame(timeTakenData, index=None, columns=['exp_num', 'min_leaf', 'Time'])

    av_time = timeTaken_pd.groupby('min_leaf').mean()
    av_cv = results_pd.groupby('min_leaf').mean()

    del av_cv['exp_num']
    del av_time['exp_num']
    return av_cv, av_time


#############################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", help="options: nn, svc, logreg, rf, ripper", default="nn")
    parser.add_argument("--param", help="index of params", default=0, type=int)
    parser.add_argument("--node", help="node", default=0, type=int)
    parser.add_argument("--runIndex", help="runIndex", default=0, type=int)
    parser.add_argument("--timeout", type=int, help="cpu timeout in seconds", default=2000)
    parser.add_argument("--dataset", help="dataset")

    args = parser.parse_args()
    # timeout = 1000
    cl_name = args.classifier
    runIndex = args.runIndex
    dataset = args.dataset

    param_index = args.param
    node = args.node


    if (cl_name == 'ripper'):
        av_cv, av_time= run_weka_RIPPER(
                                    cl_name, dataset, param_index, runIndex, node)
    else:
        av_cv, av_time=run_all_classifiers_cv(cl_name, dataset, node, runIndex, param_index)





    if (cl_name == "ripper"):
        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + cl_name + " " \
              + str(dataset) + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str(av_cv.iloc[0]['RuleSize']) + " " \
              + str("NAN") + " " \
              + str(av_time.iloc[0]['Time']) + " " \
              + str(av_cv.iloc[0]['tr_acc'] * 100.0) + " " \
              + str(0) + " " \
              + str(av_cv.iloc[0]['acc'] * 100.0) + " " \
              + str(runIndex) + " " + str(param_index) + " " + str("NAN NAN NAN")
    else:
        cmd = "python ../output/dump_result.py ../output/result.csv " \
              + cl_name + " " \
              + str(dataset) + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str("NAN") + " " \
              + str(av_time.iloc[0]['Time']) + " " \
              + str(av_cv.iloc[0]['tr_acc'] * 100.0) + " " \
              + str(0) + " " \
              + str(av_cv.iloc[0]['acc'] * 100.0) + " " \
              + str(runIndex) + " " + str(param_index) + " " + str("NAN NAN NAN")

    os.system(cmd)

    if (not True):
        print("dataset:                  " + dataset)
        print("classifier:               " + cl_name)
        print("train time:               " + str(av_time.iloc[0]['Time']))
        print("train accuracy:           " + str(av_cv.iloc[0]['tr_acc'] * 100.0))
        print("test accuracy:            " + str(av_cv.iloc[0]['acc'] * 100.0))
        # print("test accuracy:            " + str(hold_acc * 100.0))

    # os.system("rm -R ../Benchmarks/tempfiles*")


if __name__ == "__main__":
    main()
