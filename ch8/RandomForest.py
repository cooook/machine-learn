from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def rand_row(dataset, label, size):
    dataMatrix = np.column_stack((dataset, label))
    row = np.random.choice(size, size)
    TrainMatrix = dataMatrix[row,:]

    data, label = np.hsplit(TrainMatrix, (dataMatrix.shape[1] - 1,))
    return data, label


def Train(dataset, label, test_set, test_label, T, print_cnt, save=False):
    test_len = len(test_set)
    train_len = len(dataset)
    Result = np.zeros(test_len)
    global best_AUC

    Train_data, Train_label = rand_row(dataset, label, train_len)

    AUC = []
    if save:
        clf_list = []
    for i in range(T):
        clf = tree.DecisionTreeClassifier(max_features=3)
        clf.fit(Train_data, Train_label)
        if save:
            clf_list.append(clf)
        test_pred = clf.predict(test_set)
        Result += test_pred
        if i % print_cnt == 0:
            FinalResult = [1 if x >= i / 2 else 0 for x in Result]
            # FinalResult = Result
            AUC_Result = metrics.roc_auc_score(test_label, FinalResult)
            if AUC_Result > best_AUC:
                best_AUC = AUC_Result
            AUC.append(AUC_Result)
        Train_data, Train_label = rand_row(dataset, label, test_len)
    if save:
        return clf_list
    return AUC





if __name__ == '__main__':
    dataset = np.loadtxt(".\\adult_train_feature.txt")
    label = np.loadtxt(".\\adult_train_label.txt")
    test_set = np.loadtxt(".\\adult_test_feature.txt")
    test_label = np.loadtxt(".\\adult_test_label.txt")

    best_T = 4
    Total_T = 100
    print_cnt = 10
    test_cnt = 5
    best_AUC = 0


    X = [x for x in range(0, Total_T, print_cnt)]
    y = np.zeros(len(X))


    for i in range(test_cnt):
        train_X, test_X, train_y, test_y = train_test_split(dataset, label, test_size=0.2)
        tmp = Train(train_X, train_y, test_X, test_y, Total_T, print_cnt)
        y += tmp
    y = y / test_cnt

    clf_list = Train(dataset, label, test_set, test_label, best_T, print_cnt, True)
    FinalResult = np.zeros(len(test_set))
    for x in clf_list:
        FinalResult += x.predict(test_set)
    FinalResult = [1 if x >= best_T / 2 else 0 for x in FinalResult]

    # best_AUC = metrics.roc_auc_score(test_label, FinalResult)
    print('Best_AUC : %f' % best_AUC)


    plt.figure(figsize=(6, 6))
    plt.plot(X, y, color="red", linewidth=1)
    plt.xlabel("Num")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel("AUC")
    plt.ylim(0.6, 0.85)  # xlim、ylim：分别设置X、Y轴的显示范围。
    plt.xlim(0, Total_T)  # xlim、ylim：分别设置X、Y轴的显示范围。
    plt.show()
