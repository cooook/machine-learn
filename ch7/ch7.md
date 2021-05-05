# 1
## 1.
$$y=0$$
## 2.
$$P(y=0)=\frac{6}{15}=\frac{2}{5}$$
$$P(y=1)=\frac{9}{15}=\frac{3}{5}$$
$$P(x_1=0|y=0)=\frac{2}{6}$$
$$P(x_1=0|y=1)=\frac{1}{3}$$
$$P(x_2=B|y=0)=\frac{1}{2}$$
$$P(x_2=B|y=1)=\frac{1}{9}$$
$$P(y=0)\times P(x_1=0|y=0)\times P(x_2=B|y=0)\approx0.0666667$$
$$P(y=1)\times P(x_1=0|y=1)\times P(x_2=B|y=1)\approx0.0222222$$
所以y=0
## 3.
$$P(y=0)=\frac{6+1}{15+2}=\frac{7}{17}$$
$$P(y=1)=\frac{9+1}{15+2}=\frac{10}{17}$$
$$P(x_1=0|y=0)=\frac{2+1}{6+3}=\frac{1}{3}$$
$$P(x_1=0|y=1)=\frac{3+1}{9+3}=\frac{1}{3}$$
$$P(x_2=B|y=0)=\frac{3+1}{6+3}=\frac{4}{9}$$
$$P(x_2=B|y=1)=\frac{1+1}{9+3}=\frac{1}{6}$$
$$P(y=0)\times P(x_1=0|y=0)\times P(x_2=B|y=0)\approx0.061002178$$
$$P(y=1)\times P(x_1=0|y=1)\times P(x_2=B|y=1)\approx0.032679738$$
$$y=0$$
# 2
![截屏2021-05-04 上午8.44.54.png](https://i.loli.net/2021/05/04/DClYbG7FzkBoXdW.png)

$$Pr(A, B, C, D, E, F) = Pr(A)Pr(B)Pr(C|A, B)Pr(D|B)Pr(E|C,D)Pr(F|E)$$
道德图如下所示
![截屏2021-05-04 上午9.00.02.png](https://i.loli.net/2021/05/04/h8uQDmZiNj7v1cB.png)

![图片1.png](https://i.loli.net/2021/05/04/J7fY89rUSP6Gqey.png)

# 3

```python
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
```

```
7.722360621178051e-05 0.025631024529740677
好瓜
```

