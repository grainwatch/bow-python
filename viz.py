import cv2 as cv
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import dataset
import model


id_to_classname_dict = {-1: 'Mehl', 1: 'Mais', 2: 'Roggen', 3: 'Triticale', 4: 'Weizen'}

def visualize(x, y): 
    c = np.asarray(y).astype(np.float)
    pca = decomposition.PCA(3)
    x = pca.fit_transform(x)
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, cmap=cm.get_cmap('gist_rainbow'))
    handles, labels = scatter.legend_elements()
    labels = id_to_classname_dict.values()
    ax.legend(handles, labels, title="Klassen")
    plt.show()

def _id_to_classname(masked_array):
    id_list = masked_array.astype(np.int).tolist()
    classname_list = list(map(lambda x: id_to_classname_dict[x], id_list))
    return np.asarray(classname_list)
    
    

if __name__ == "__main__":
    #x, y = dataset.load_2_class('C:/Users/Alex/CodeProjects/bow-python/dataset', -1, 1)
    train, test = dataset.load_data_2('C:/Users/Alex/CodeProjects/bow-python/dataset_v2')
    #(train_x, train_y), (test_x, test_y) = dataset.filter_data(train, test, [-1, 1])
    train_x, train_y = train
    test_x, test_y = test
    train_x, train_y = dataset.shuffle(train_x, train_y)
    """bow = model.BOWModel(150)
    bow.fit(train_x, train_y)
    visualize(bow.x_histogram, train_y)"""
    hog = model.HOG()
    hog.fit(train_x, train_y)
    visualize(hog.x_des, train_y)
