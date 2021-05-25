from sklearn.decomposition import PCA
import os
import numpy as np
from PIL import Image

def loadImage(path):
    img = Image.open(path)
    img = img.convert("L")
    data = img.getdata()
    data = np.array(data).reshape(1, -1)/100
    return data



path = '/Users/cooook/Desktop/machine learn/ch10/yalefaces/'

files = os.listdir(path)

img = []
pca = PCA(n_components=20)


