from sklearn import tree, metrics
import numpy as np
import matplotlib.pyplot as plt


def AdaBoost(dataset, label, test_set, test_label, T, cnt):
    clf = tree.DecisionTreeClassifier(max_depth=1, random_state=1)
    train_size = len(dataset)
    test_size = len(test_set)
    w = np.ones(train_size) / train_size
    Result = np.zeros(test_size)
    for i in range(T):
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
            Times.append(i)
            FinalResult = np.sign(Result)
            AUC.append(metrics.roc_auc_score(test_label, FinalResult))
        Z = 2 * np.sqrt(error_rate * (1 - error_rate))
        w = w * ([np.exp(-alpha * x) for x in error_two]) / Z

def fromZerotoOne(x):
    if x == 1:
        return 1
    return -1

if __name__ == '__main__':
    dataset = np.loadtxt("./ch8/adult_train_feature.txt")
    label = np.loadtxt("./ch8/adult_train_label.txt")
    test_set = np.loadtxt("./ch8/adult_test_feature.txt")
    test_label = np.loadtxt("./ch8/adult_test_label.txt")
    Times = []
    AUC = []

    label = np.array(list(map(fromZerotoOne, label)))
    test_label = np.array(list(map(fromZerotoOne, test_label)))

    AdaBoost(dataset, label, test_set, test_label, 500, 10)

    plt.figure(figsize=(6, 6))
    plt.plot(Times, AUC, color="red", linewidth=1)
    plt.xlabel("Num")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel("AUC")
    plt.ylim(0.6, 0.85)  # xlim、ylim：分别设置X、Y轴的显示范围。
    plt.show()
