from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def AdaBoost(dataset, label, test_set, test_label, T, cnt):
    clf = tree.DecisionTreeClassifier(max_depth=2)
    train_size = len(dataset)
    test_size = len(test_set)
    w = np.ones(train_size) / train_size
    Result = np.zeros(test_size)
    AUC = []
    global best_AUC, best_T
    for i in range(1, T + 1):
        clf.fit(dataset, label, sample_weight=w)
        train_pred = np.array(clf.predict(dataset))
        test_pred = np.array(clf.predict(test_set))
        error_list = list(map(int, train_pred != label))
        error_two = [-1 if x == 1 else 1 for x in error_list]
        error_rate = sum(error_list * w)
        if error_rate > 0.5:
            break
        alpha = 0.5 * np.log((1 - error_rate) / error_rate)
        Result = Result + alpha * test_pred
        if i % cnt == 0:
            FinalResult = np.sign(Result)
            Tmp_AUC = metrics.roc_auc_score(test_label, FinalResult)
            if Tmp_AUC > best_AUC:
                best_AUC = Tmp_AUC
                best_T = i
            AUC.append(metrics.roc_auc_score(test_label, FinalResult))
        Z = 2 * np.sqrt(error_rate * (1 - error_rate))
        w = w * ([np.exp(-alpha * x) for x in error_two]) / Z
    return AUC

def fromZerotoOne(x):
    if x == 1:
        return 1
    return -1

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

    label = np.array(list(map(fromZerotoOne, label)))
    test_label = np.array(list(map(fromZerotoOne, test_label)))

    # AdaBoost(dataset, label, test_set, test_label, 500, 10)
    X = [x for x in range(print_cnt, Total_T + 1, print_cnt)]
    y = np.zeros(len(X))

    for i in range(test_cnt):
        train_X, test_X, train_y, test_y = train_test_split(dataset, label, test_size=0.2)
        tmp = AdaBoost(train_X, train_y, test_X, test_y, Total_T, print_cnt)
        y += tmp
    y = y / test_cnt
    print(best_T)

    plt.figure(figsize=(6, 6))
    plt.plot(X, y, color="red", linewidth=1)
    plt.xlabel("Num")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel("AUC")
    plt.ylim(0.6, 0.85)  # xlim、ylim：分别设置X、Y轴的显示范围。
    plt.show()

    AUC = AdaBoost(dataset, label, test_set, test_label, best_T, best_T)[0]
    print(AUC)


