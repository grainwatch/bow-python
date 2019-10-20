import numpy as np
import cv2 as cv
import sklearn.metrics as metrics

class HOGModel:
    def __init__(self, win_size=(128, 128), blocks=(2, 2), cells_per_block=(2, 2), block_stride_in_cells=(1, 1), bins=9):
        block_size = (win_size[0]//blocks[0], win_size[1]//blocks[0])
        cell_size = (block_size[0]//cells_per_block[0], block_size[1]//cells_per_block[1])
        block_stride_size = cell_size
        self.hog = cv.HOGDescriptor(win_size, block_size, block_stride_size, cell_size, bins)
        self.svm = cv.ml.SVM_create()
        self.x_des = None
        self.fitted = False
    
    def fit(self, x, y):
        if self.fitted:
            self.svm = cv.ml.SVM_create()
        self.x_des = list(map(lambda img: self.hog.compute(img), x))
        self.x_des = np.asarray(self.x_des, np.float32)
        self.x_des = self.x_des.reshape((self.x_des.shape[0], self.x_des.shape[1]))
        y = np.asarray(y, np.int32)
        self.svm.trainAuto(self.x_des, cv.ml.ROW_SAMPLE, y)
        self.fitted = True
        #self.hog.setSVMDetector(self.svm.getSupportVectors())

    def predict(self, x):
        x_des = list(map(lambda img: self.hog.compute(img), x))
        x_des = np.asarray(x_des, np.float32)
        result, response = self.svm.predict(x_des)
        response = response.astype(np.int32).reshape(response.shape[0])
        return response

    def evaluate(self, x, y):
        y = np.asarray(y)
        response = self.predict(x)
        accuracy = metrics.accuracy_score(y, response)
        return accuracy

    def save(self, hog_filename, svm_filename):
        self.hog.save(hog_filename)
        self.svm.save(svm_filename)

    @staticmethod
    def load(hog_filename, svm_filename):
        hog = cv.HOGDescriptor()
        hog.load(hog_filename)
        svm = cv.ml.SVM_load(svm_filename)
        hog_model = HOG()
        hog_model.hog = hog
        hog_model.svm = svm
        return hog_model