import cv2 as cv
import numpy as np
import dataset
import random

class BOWModel:
    def __init__(self, clusters):
        self.clusters = clusters
        self.svm = cv.ml.SVM_create()
        self.bow_trainer = cv.BOWKMeansTrainer(clusters)
        self.matcher = cv.FlannBasedMatcher_create()
        self.ft_ext = cv.xfeatures2d.SIFT_create()
        self.x_ft = None
        self.x_histogram = None
        self.vocabulary = None
        self.is_fitted = False

    def fit(self, x, y):
        self.x = x
        self.x_ft = list(map(self._compute_ft, x))
        for ft in self.x_ft:
            self.bow_trainer.add(ft)
        self.vocabulary = self.bow_trainer.cluster()
        self.matcher.add([self.vocabulary])
        self.x_histogram = list(map(self._compute_histogram, self.x_ft))
        self.x_histogram = np.asarray(self.x_histogram, np.float32)
        y_array = np.asarray(y, np.int32)
        self.svm.trainAuto(self.x_histogram, cv.ml.ROW_SAMPLE, y_array)
        self.is_fitted = True

    def _compute_ft(self, x_item):
        kp, ds = self.ft_ext.detectAndCompute(x_item, None)
        return ds

    def _compute_histogram(self, x_item):
        histogram = [0] * self.clusters
        matches = self.matcher.match(x_item)
        for match in matches:
            histogram[match.trainIdx] += 1
        return histogram  

    def predict(self, x):
        if self.is_fitted:
            x_ft = map(self._compute_ft, x)
            x_histogram = map(self._compute_histogram, x_ft)
            x_histogram = np.asarray(list(x_histogram), np.float32)
            return self.svm.predict(x_histogram)
    
    def evaluate(self, x, y):
        y = np.asarray(y)
        result, response = self.predict(x)
        response = response.astype(np.int).reshape(response.shape[0])
        accuracy = 0
        equal = np.equal(response, y, dtype=np.int32)
        print(equal)
        equal_corn = []
        for i in range(equal.shape[0]):
            if y[i] == 1:
                equal_corn.append(equal[i])
        equal_count = np.sum(equal)
        accuracy = equal_count/response.shape[0]
        corn_accuracy = np.sum(np.asarray(equal_corn))/len(equal_corn)
        return accuracy


class HOG:
    def __init__(self, win_size=(128, 128), blocks=(2, 2), cells_per_block=(2, 2), block_stride_in_cells=(1, 1), bins=9):
        block_size = (win_size[0]//blocks[0], win_size[1]//blocks[0])
        cell_size = (block_size[0]//cells_per_block[0], block_size[1]//cells_per_block[1])
        block_stride_size = cell_size
        self.hog = cv.HOGDescriptor(win_size, block_size, block_stride_size, cell_size, bins)
        self.svm = cv.ml.SVM_create()
    
    def fit(self, x, y):
        x_des = list(map(lambda img: self.hog.compute(img), x))
        x_des = np.asarray(x_des, np.float32)
        y = np.asarray(y, np.int32)
        self.svm.trainAuto(x_des, cv.ml.ROW_SAMPLE, y)
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
        accuracy = _calcute_array_equal_ratio(y, response)
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
        

class ObjectDetector:
    def __init__(self, detect_model, win_size, win_stride):
        self.detect_model = detect_model
        self.win_size = win_size
        self.win_stride = win_stride
    
    def detect(self, x, true_class=1):
        windows_horizontal = (x.shape[1] - self.win_size[0])//self.win_stride[0] + 1
        windows_vertical = (x.shape[0] - self.win_size[1])//self.win_stride[1] + 1
        win_pos = [0, 0]
        roi_list = []
        roi_pos_list = []
        while win_pos[1] + self.win_size[1] < x.shape[0]:
            roi = x[win_pos[1]:win_pos[1]+self.win_size[1], win_pos[0]:win_pos[0]+self.win_size[0]]
            roi_list.append(roi)
            roi_pos_list.append(win_pos.copy())
            win_pos[0] += self.win_stride[0]
            if win_pos[0] + self.win_size[0] > x.shape[1]:
                win_pos[0] = 0
                win_pos[1] += self.win_stride[1]
        response = self.detect_model.predict(roi_list)
        true_array = np.full(len(roi_pos_list), true_class)
        equal_array = np.equal(true_array, response)
        roi_array = np.asarray(roi_pos_list)
        shape = roi_array.shape
        roi_array = np.compress(equal_array, roi_pos_list)
        
        return roi_array
        


class ColourDBScan:
    pass

def _calcute_array_equal_ratio(array_true, array_to_check):
    accuracy = 0
    equal = np.equal(array_true, array_to_check, dtype=np.int32)
    equal_count = np.sum(equal)
    accuracy = equal_count/array_to_check.shape[0]
    return accuracy

def _find_elements(array, element):
    indices = []
    for i in range(array.shape[0]):
        if array[i] == element:
            indices.append(i)
    return indices

def _filter_by_index_list(array, indices):
    filtered = []
    for i in indices:
        array[i]


if __name__ == "__main__":
    """bow = BOWModel(150)
    hog = HOG((128, 128), (2, 2), (2, 2), (1, 1), 9)
    train, test = dataset.load_data_json('C:/Users/Alex/CodeProjects/bow-python/dataset')
    (train_x, train_y), (test_x, test_y) = dataset.filter_data(train, test, [-1, 1])
    print('filtered')
    train_x, train_y = dataset.shuffle(train_x, train_y)
    print('shuffled')
    #print(train_x, train_y, len(train_x))
    bow.fit(train_x, train_y)
    hog.fit(train_x, train_y)
    print('trained')
    accuracy = bow.evaluate(test_x, test_y)
    print(f'accuracy bow: {accuracy}')
    accuracy = hog.evaluate(test_x, test_y)
    print(f'accuracy hog: {accuracy}')"""
    hog = HOG()
    train, test = dataset.load_data_json('C:/Users/Alex/CodeProjects/bow-python/dataset')
    (train_x, train_y), (test_x, test_y) = dataset.filter_data(train, test, [-1, 1])
    train_x, train_y = dataset.shuffle(train_x, train_y)
    hog.fit(train_x, train_y)
    accuracy = hog.evaluate(test_x, test_y)
    print(f'accuracy hog: {accuracy}')
    hog.save('hog', 'svm')



