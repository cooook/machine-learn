import numpy as np

Train_data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103]])
Train_Size = Train_data.shape[0]

Train_label = np.array(['是', '是', '是', '是', '是', '是', '是', '是', '否', '否', '否', '否','否', '否', '否', '否', '否'])

def dis(x, y):
    return np.linalg.norm(x - y)

def KNN(test_input, K):
    Dis_List = [(dis(Train_data[i], test_input), Train_label[i]) for i in range(Train_Size)]
    Dis_List = sorted(Dis_List, key=lambda x: x[0])
    Ans = 0
    for i in range(K):
        Ans += 1 if Dis_List[i][1] == '是' else -1
    return '是' if Ans >= 0 else '否'


if __name__ == '__main__':
    Result = KNN([0.5, 0.5], 5)
    print(Result)
