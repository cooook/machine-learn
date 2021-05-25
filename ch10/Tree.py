import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap

Train_data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103]])
Train_Size = Train_data.shape[0]

Train_label = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def plot_decision_boundary(clf, axes):
    xp = np.linspace(axes[0], axes[1], 100)
    yp = np.linspace(axes[2], axes[3], 100)
    x1, y1 = np.meshgrid(xp, yp)
    xy = np.c_[x1.ravel(), y1.ravel()]
    y_pred = clf.predict(xy).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, y1, y_pred, alpha=0.3, cmap=custom_cmap)

if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier()
    clf.fit(Train_data, Train_label)
    plot_decision_boundary(clf, axes=[0, 1, 0, 1])
    p1 = plt.scatter(Train_data[Train_label == 1, 0], Train_data[Train_label == 1, 1], color='blue')
    p2 = plt.scatter(Train_data[Train_label == 0, 0], Train_data[Train_label == 0, 1], color='green')
    plt.legend([p1, p2], ['Good melon', 'bad melon'], loc='upper right')
    plt.show()

