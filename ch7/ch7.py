from pandas import read_csv
import numpy as np

file_name = "~/Desktop/machine learn/ch7/data3.0.csv"
test_name = "~/Desktop/machine learn/ch7/test.csv"

dataset = read_csv(file_name)
test_input = read_csv(test_name)


def pn(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x-mu) ** 2 / (2 * sigma ** 2))

p_1 = (len(dataset[dataset['好瓜'] == '是']) + 1) / (len(dataset) + len(dataset['好瓜'].unique())) # 好瓜
p_0 = (len(dataset[dataset['好瓜'] == '否']) + 1) / (len(dataset) + len(dataset['好瓜'].unique())) # 坏瓜
attr = dataset.columns.values


test_array = np.array(test_input)
for i in test_array:
    for x in range(1, len(i)):
        attr_x = i[x]
        if type(attr_x) == float:
            mu = dataset[(dataset['好瓜'] == '否')]
            mu = np.float64(mu[[attr[x]]].mean())
            sigma = dataset[(dataset['好瓜'] == '否')]
            sigma = np.sqrt(np.float64(sigma[attr[x]].var()))
            p_0 *= pn(attr_x, mu, sigma)

            mu = dataset[(dataset['好瓜'] == '是')]
            mu = np.float64(mu[[attr[x]]].mean())
            sigma = dataset[(dataset['好瓜'] == '是')]
            sigma = np.sqrt(np.float64(sigma[attr[x]].var()))
            p_1 *= pn(attr_x, mu, sigma)
        else:
            p_1 *= (len(dataset[(dataset['好瓜'] == '是') & (dataset[attr[x]] == attr_x)]) + 1) / \
                   (len(dataset[dataset['好瓜'] == '是']) + len(dataset[attr[x]].unique()))
            p_0 *= (len(dataset[(dataset['好瓜'] == '否') & (dataset[attr[x]] == attr_x)]) + 1) / \
                   (len(dataset[dataset['好瓜'] == '否']) + len(dataset[attr[x]].unique()))
print(p_0, p_1)
if p_0 > p_1:
    print('坏瓜')
else:
    print('好瓜')