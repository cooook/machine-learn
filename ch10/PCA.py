from sklearn.decomposition import PCA
import os
import numpy as np
from PIL import Image

def loadImage(path):
    img = Image.open(path)
    img = img.convert("L")
    data = img.getdata()
    data = np.array(data).reshape(1, -1)[0]
    return data



path = 'D:\\zyf\\machine-learn\\ch10\\yalefaces\\'

files = os.listdir(path)

img = []
pca = PCA(n_components=20)
for file in files:
    if file != '.DS_Store':
        img.append(loadImage(path + file))
img = np.array(img)
newImg = pca.fit_transform(img)
Matrix = pca.components_
# newImg = pca.transform(img)
for i in range(len(newImg)):
    if files[i] == '.DS_Store':
        continue
    image = newImg[i]
    tmp = np.dot(image, Matrix)
    tmp = tmp.reshape(243, 320)
    tmp += np.mean(img, axis=0).reshape(243, 320)
    new_img = Image.fromarray(tmp.astype('uint8'))
    new_img = new_img.convert("L")
    new_img.save('D:\\zyf\\machine-learn\\ch10\\mydata\\{0}.png'.format(files[i]))
