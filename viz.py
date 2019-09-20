import cv2 as cv
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import dataset
import bow_model

def visualize(x, y, colors):
    c = list(map(lambda x: colors[x], y))
    c = np.asarray(c)
    print(c)
    pca = decomposition.PCA(3)
    x = pca.fit_transform(x)
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c)
    plt.show()

if __name__ == "__main__":
    #x, y = dataset.load_2_class('C:/Users/Alex/CodeProjects/bow-python/dataset', -1, 1)
    (train_x, train_y), (test_x, test_y) = dataset.load_data_json('C:/Users/Alex/CodeProjects/bow-python/dataset')
    x, y = dataset.shuffle(train_x, train_y)
    x, y = dataset.filter_data(x, y, )
    bow = bow_model.BOWModel(150)
    bow.fit(x, y)
    print(bow.x_histogram.shape)
    colors = {-1: '#ff0000', 1: '#00ff00', 2: '#00ffff', 3: '#0000ff', 4: '#ff00ff'}
    visualize(bow.x_histogram, y, colors)
