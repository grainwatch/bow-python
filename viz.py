import cv2 as cv
import numpy as np
import dataclasses
from sklearn import decomposition
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from util.rectangle import Rect
import dataset
import models


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

def viz_detected_objects(img: np.ndarray, true_rects: List[Rect], true_labels, detected_rects: List[Rect], detected_labels):
    for rect, label in zip(true_rects, true_labels):
        img = cv.rectangle(img, dataclasses.astuple(rect), (0, 255, 0), 3)
        img = cv.putText(img, f'{label}', (rect.x, rect.y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    for rect, label in zip(detected_rects, detected_labels):
        img = cv.rectangle(img, dataclasses.astuple(rect), (0, 0, 255), 3)
        img = cv.putText(img, f'{label}', (rect.x, rect.y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    img = cv.resize(img, (0, 0), None, 0.5, 0.5)
    cv.imshow('', img)
    cv.waitKey(0)

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
    bow = models.BOWModel(150)
    bow.fit(train_x, train_y)
    visualize(bow.x_histogram, train_y)
    """hog = models.HOG()
    hog.fit(train_x, train_y)
    visualize(hog.x_des, train_y)"""