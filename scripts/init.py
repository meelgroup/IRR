import argparse
import os
from helper import discretization_from_file, save_in_disk, partition_with_eq_prob, do_k_fold, \
    partition_binary_class_data

verbose = not True
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--partition", type=int, help="partition", default=8)
    parser.add_argument("--kFold", type=int, help="kFold", default=5)
    parser.add_argument("--threshold", type=int, help="threshold", default=9)

    args = parser.parse_args()
    dataset = args.dataset
    partition_count = args.partition
    kFold = args.kFold
    numFresh = args.threshold

    location = "../Benchmarks/Data/"
    datasets = ["adult_data_bintarget.csv",
                "credit_card_clients.csv",
                "ilpd.csv",
                "ionosphere.csv",
                "iris_bintarget.csv",
                "parkinsons.csv",
                "pima-indians-diabetes.csv",
                "TomsHardware_bintarget.csv",
                "transfusion.csv",
                "Twitter_bintarget_small.csv",
                "wdbc.csv"]

    columnSeperator = ","
    rowHeader = 0
    columnY = None
    fracPresent = 0.9
    columnsCategorical = []
    valEqY = None

    if (dataset == "ion"):
        file = open(location + datasets[3], "r")
        fileName = location + datasets[3]
        columnNames = file.readline()[:-1].split(columnSeperator)
    #
    elif (dataset == "credit"):
        file = open(location + datasets[1], "r")
        fileName = location + datasets[1]
        columnNames = file.readline()[:-1].split(columnSeperator)
        columnsCategoricalIndex = [2, 3, 4]
        for each_index in columnsCategoricalIndex:
            columnsCategorical.append(columnNames[each_index])
        if (verbose):
            print(columnsCategorical)

    elif (dataset == "adult"):
        file = open(location + datasets[0], "r")
        fileName = location + datasets[0]
        columnNames = file.readline()[:-1].split(columnSeperator)
        columnsCategoricalIndex = [2, 4, 6, 7, 8, 9, 10, 14, 15]
        for each_index in columnsCategoricalIndex:
            columnsCategorical.append(columnNames[each_index])
        if (verbose):
            print(columnsCategorical)
    if (dataset == "iris"):
        file = open(location + datasets[4], "r")
        fileName = location + datasets[4]
        columnNames = file.readline()[:-1].split(columnSeperator)[:]
        file.close()
    #
    # # already done
    #
    if (dataset == "twitter"):
        file = open(location + datasets[9], "r")
        fileName = location + datasets[9]
        columnNames = file.readline()[:-1].split(columnSeperator)
    #
    elif (dataset == "parkinsons"):
        file = open(location + datasets[5], "r")
        fileName = location + datasets[5]
        columnNames = file.readline()[:-1].split(columnSeperator)[1:]
    #
    if (dataset == "pima"):
        file = open(location + datasets[6], "r")
        fileName = location + datasets[6]
        columnNames = file.readline()[:-1].split(columnSeperator)[1:]

    if (dataset == "wdbc"):
        file = open(location + datasets[10], "r")
        fileName = location + datasets[10]
        columnNames = file.readline()[:-1].split(columnSeperator)[1:]

    if (dataset == "trans"):
        file = open(location + datasets[8], "r")
        fileName = location + datasets[8]
        columnNames = file.readline()[:-1].split(columnSeperator)[1:]

    if (dataset == "toms"):
        file = open(location + datasets[7], "r")
        fileName = location + datasets[7]
        columnNames = file.readline()[:-1].split(columnSeperator)[1:]
    print("\n\n" + dataset + "\n\n")
    X, y, columnInfo = discretization_from_file(fileName, columnSeperator, rowHeader, columnNames, columnY, True,
                                                fracPresent,
                                                columnsCategorical, numFresh, valEqY)
    print(X.shape, y.shape,)
    save_location = "../Benchmarks/tempfiles/"
    os.system("mkdir " + save_location)

    '''
        do k fold and save dataset instances seperately
    '''

    XTrains, yTrains, XTests, yTests = do_k_fold(X, y, kFold)

    for i in range(kFold):
        save_in_disk(XTrains[i], yTrains[i], save_location, dataset + "_train_" + str(i) + ".data", columnInfo)
        save_in_disk(XTests[i], yTests[i], save_location, dataset + "_test_" + str(i) + ".data", columnInfo)

    # save in arff format for ripper
    for i in range(kFold):
        XTrains[i].columns = [
            str(a.replace(" ", "_")) + str("_") + str(b.replace(" ", "_")) + str("_") + str(c.replace(" ", "_"))
            for
            (a, b, c) in
            XTrains[i].columns]
        yTrains[i] = yTrains[i].astype(int)
        yTests[i] = yTests[i].astype(int)

        XTrains[i]["target"] = yTrains[i]
        file = open(save_location + str(dataset) + "_train_" + str(i) + ".arff", "w")
        file.write("@relation " + str(dataset))
        file.write("\n\n")
        for column in XTrains[i].columns:
            file.write("@attribute " + str(column) + str(" {0,1}\n"))
        file.write("\n\n@data\n")
        XTrains[i].to_csv(file, index=False, header=False)
        file.close()

        XTests[i].columns = [
            str(a.replace(" ", "_")) + str("_") + str(b.replace(" ", "_")) + str("_") + str(c.replace(" ", "_"))
            for
            (a, b, c) in
            XTests[i].columns]
        XTests[i]["target"] = yTests[i]
        file = open(save_location + str(dataset) + "_test_" + str(i) + ".arff", "w")
        file.write("@relation " + str(dataset))
        file.write("\n\n")
        for column in XTests[i].columns:
            file.write("@attribute " + str(column) + str(" {0,1}\n"))
        file.write("\n\n@data\n")
        XTests[i].to_csv(file, index=False, header=False)
        file.close()
