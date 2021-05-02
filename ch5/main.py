import numpy as np
import pandas as pd
from math import fabs
from random import random

# sigmoid
def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)）
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度）
    return sigmoid(x) * (1 - sigmoid(x))


# loss
def mse_loss(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差）,其中真实标记和预测值都是长度相同的向量）

    return 0.5 * np.linalg.norm(y_true - y_pred) ** 2


class NeuralNetwork_221():
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # 以上为神经网络中的变量，其中具体含义见网络图

    def predict(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] - self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] - self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 - self.b3)
        return o1

    def FromSelfToArray(self, w):
        w = [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.b1, self.b2, self.b3]

    def FromArrayToSelf(self, w):
        self.w1 = w[0]
        self.w2 = w[1]
        self.w3 = w[2]
        self.w4 = w[3]
        self.w5 = w[4]
        self.w6 = w[5]
        self.b1 = w[6]
        self.b2 = w[7]
        self.b3 = w[8]

    def train_1(self, data, all_y_trues):
        # learn_rate = 0.1
        epochs = 500
        last_loss = 0
        eps = 1e-7
        Delta = 0.95
        T = 10
        w = [0 for i in range(9)]
        self.FromSelfToArray(w)
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 以下部分为向前传播过程，请完成
                learn_rate = T * random()
                sum_h1 = x[0] * self.w1 + x[1] * self.w2  # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = x[0] * self.w3 + x[1] * self.w4  # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_ol = h1 * self.w5 + h2 * self.w6  # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol - self.b3)  # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = y_pred - y_true  # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = h1 * y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = h2 * y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = -y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = y_pred * (1 - y_pred) * self.w5  # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = y_pred * (1 - y_pred) * self.w6  # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = -deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = -deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5  # （需要填写的地方，更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6  # （需要填写的地方，更新w6）
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3  # （需要填写的地方，更新b3）

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1  # （需要填写的地方，更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2  # （需要填写的地方，更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1  # （需要填写的地方，更新b1）

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3  # （需要填写的地方，更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4  # （需要填写的地方，更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2  # （需要填写的地方，更新b2）

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                if loss > last_loss and random() < np.exp((last_loss - loss) / T):
                    self.FromArrayToSelf(w)
                else:
                    self.FromSelfToArray(w)
                last_loss = loss
            T *= Delta
            if T < eps:
                break

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        last_loss = 0
        eps = 0.01
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 以下部分为向前传播过程，请完成
                sum_h1 = x[0] * self.w1 + x[1] * self.w2  # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = x[0] * self.w3 + x[1] * self.w4  # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_ol = h1 * self.w5 + h2 * self.w6  # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol - self.b3)  # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = y_pred - y_true  # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = h1 * y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = h2 * y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = -y_pred * (1 - y_pred)  # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = y_pred * (1 - y_pred) * self.w5  # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = y_pred * (1 - y_pred) * self.w6  # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = -deriv_sigmoid(sum_h1 - self.b1)  # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = -deriv_sigmoid(sum_h2 - self.b2)  # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5  # （需要填写的地方，更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6  # （需要填写的地方，更新w6）
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3  # （需要填写的地方，更新b3）

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1  # （需要填写的地方，更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2  # （需要填写的地方，更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1  # （需要填写的地方，更新b1）

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3  # （需要填写的地方，更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4  # （需要填写的地方，更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2  # （需要填写的地方，更新b2）
            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                if fabs(last_loss - loss) < eps:
                    return
                last_loss = loss


def main():
    X_train = np.genfromtxt('/Users/cooook/Desktop/machine learn/ch5/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('/Users/cooook/Desktop/machine learn/ch5/train_target.csv', delimiter=',')
    X_test = np.genfromtxt('/Users/cooook/Desktop/machine learn/ch5/test_feature.csv', delimiter=',')  # 读取测试样本特征
    network = NeuralNetwork_221()
    network.train_1(X_train, y_train)
    y_pred = []
    for i in X_test:
        y_pred.append(int(network.predict(i) > 0.5))  # 将预测值存入y_pred(list)内
    y_pd = pd.DataFrame(y_pred)
    y_pd.to_csv('/Users/cooook/Desktop/machine learn/ch5/test_target.csv', index=False)
    ##############
    # （需要填写的地方，选定阈值，将输出对率结果转化为预测结果并输出）
    ##############


main()
