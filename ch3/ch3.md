# 1
## (1)
带入得$(b-w)^2+(b)^2+(1-(w+b))^2$  
$=(b^2-2bw+w^2) + b^2 + 1 - 2(w + b) + w^2 + 2wb + b^2$
$=3b^2+2w^2-2w-2b+1$
所以 $w = \frac{1}{2}, b = \frac{1}{3}$  

## (2)
$dis^2 = \sum{\frac{(w_Exi+b_E+y_i)^2}{w_E^2+1}}$
$=\frac{b_E^2}{w_E^2+1}+\frac{(b_E-w_E)^2}{w_E^2+1}+\frac{(1+b_E+w_E)^2}{w_E^2+1}$
$=\frac{3b_E^2+2w_E^2+2b_E+2w_E+1}{w_E^2+1}$
分别对$b_E, w_E$求偏导. 
对$b$求偏导: $w\overline{x} - \overline{y} + b = 0$
$b = \frac{1}{3}$  
对$w$求偏导: $\frac{w}{m}\sum_{i=1}^{m}(x_i^2-y_i^2)+\frac{w^2-1}{m}\sum_{i = 1}^{m}x_iy_i+2bw\overline{y}-bw^2\overline{x}+b\overline{x}-b^2w=0$
带入数值得$w = \frac{\sqrt{13}}{3}-\frac{2}{3} = 0.535$


## (3)
$(w*, b*) = argmin_{w,b}{\sum_{i=1}^m{|\frac{wx_i+b-y_i}{\sqrt{w^2+1}}|}}$
当$(w*, b*) = (0.5, \frac{1}{3})时, \sum_{i=1}^m{|\frac{wx_i+b-y_i}{\sqrt{w^2+1}}|}=0.596$
而当$(w, b) = (0.5, 0.5)$时, $\sum_{i=1}^m{|\frac{wx_i+b-y_i}{\sqrt{w^2+1}}|}=0.45<0.596$ 所以$(w*, b*) != (0.5, \frac{1}{3})$

# 2
$\frac{p(y=i|x)}{p(y=K|x)}=\frac{e^{z_i}}{\sum{e^{z_i}}}$

# 3
方法一
```python
import numpy as np
from pandas import read_csv
from math import exp

theta = 0.5

dataset = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\train_feature.csv')
dataset['10'] = [1 for i in range(len(dataset))]
ground_truth = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\train_target.csv')
# print(dataset)
dataMatrix = np.array(dataset[1:])
truthMatrix = np.array(ground_truth[1:])
betaMatrix = np.dot(np.dot(np.linalg.inv(np.dot(dataMatrix.T, dataMatrix)), dataMatrix.T), truthMatrix)

val_feature = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\val_feature.csv')
val_target = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\val_target.csv')

val_feature['10'] = [1 for i in range(len(val_feature))]

valMatrix = np.array(val_feature)
val_Truth = np.array(val_target)

TP = 0
FP = 0
TN = 0
FN = 0

z = np.dot(valMatrix, betaMatrix)

for i in range(len(z)):
    f = 1 / (1 + exp(-z[i]))
    if f > theta:
        if val_Truth[i] == 1:
            TP += 1
        else:
            FP += 1
    else:
        if val_Truth[i] == 1:
            FN += 1
        else :
            TN += 1

print(betaMatrix)
print(TP, FP, FN, TN)

print('准确率: %f 查全率: %f 查准率: %f' % ((TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), TP / (TP + FP)))
```
准确率: 0.740000 查全率: 1.000000 查准率: 0.666667
方法二:
```python
import numpy as np
from pandas import read_csv
from numpy import exp

theta = 0.5

def p1(x):
    return x / (1 + x)


dataset = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\train_feature.csv')
dataset['10'] = [1 for i in range(len(dataset))]
ground_truth = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\train_target.csv')
dataMatrix = np.array(dataset[:])
truthMatrix = np.array(ground_truth[:])
val_feature = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\val_feature.csv')
val_target = read_csv('F:\\杂\\大二下\\机器学习\\ch03-data\\val_target.csv')
val_feature['10'] = [1 for i in range(len(val_feature))]
valMatrix = np.array(val_feature)
val_Truth = np.array(val_target)



betaMatrix = np.zeros((11, 1))
betaMatrix[10][0] = 1


loop = 0

while loop < 100:
    Ans = np.dot(dataMatrix, betaMatrix)
    tmp1 = np.zeros((11, 11))
    tmp2 = np.zeros((11, 1))
    tmp_Matrix = np.zeros((11, 1))

    for i in range(len(dataMatrix)):
        P1 = p1(exp(Ans[i][0]))
        tmp_Matrix = dataMatrix[i:i+1, :].T
        tmp2 -= tmp_Matrix * (truthMatrix[i][0] - P1)
        tmp1 += np.dot(tmp_Matrix, tmp_Matrix.T) * P1 * (1 - P1)

    betaMatrix -= np.dot(np.linalg.inv(tmp1), tmp2)
    loop += 1

print(betaMatrix)

TP = 0
FP = 0
TN = 0
FN = 0

z = np.dot(valMatrix, betaMatrix)

for i in range(len(z)):
    f = 1 / (1 + exp(-z[i]))
    if f > theta:
        if val_Truth[i] == 1:
            TP += 1
        else:
            FP += 1
    else:
        if val_Truth[i] == 1:
            FN += 1
        else :
            TN += 1
print(TP, FP, TN, FN)
print('准确率: %f 查全率: %f 查准率: %f' % ((TP + TN) / (TP + TN + FP + FN), TP / (TP + FN), TP / (TP + FP)))

```
准确率: 1.000000 查全率: 1.000000 查准率: 1.000000