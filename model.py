import cv2 as cv
import numpy as np
import dataset
import random
from collections import Counter

class BOWModel:
    def __init__(self, clusters=150):
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

    def save(self, bow_filename, svm_filename):
        self.matcher.save(bow_filename)
        self.svm.save(svm_filename)

    @staticmethod
    def load(bow_filename, svm_filename):
        bow = cv.FlannBasedMatcher.load(bow_filename)
        svm = cv.ml.SVM_load(svm_filename)
        bowmodel = BOWModel()
        bowmodel.matcher = bow
        bowmodel.svm = svm


class HOG:
    def __init__(self, win_size=(128, 128), blocks=(2, 2), cells_per_block=(2, 2), block_stride_in_cells=(1, 1), bins=9):
        block_size = (win_size[0]//blocks[0], win_size[1]//blocks[0])
        cell_size = (block_size[0]//cells_per_block[0], block_size[1]//cells_per_block[1])
        block_stride_size = cell_size
        self.hog = cv.HOGDescriptor(win_size, block_size, block_stride_size, cell_size, bins)
        self.svm = cv.ml.SVM_create()
        self.x_des = None
    
    def fit(self, x, y):
        print(self.svm.getSupportVectors())
        self.x_des = list(map(lambda img: self.hog.compute(img), x))
        self.x_des = np.asarray(self.x_des, np.float32)
        self.x_des = self.x_des.reshape((self.x_des.shape[0], self.x_des.shape[1]))
        y = np.asarray(y, np.int32)
        self.svm.trainAuto(self.x_des, cv.ml.ROW_SAMPLE, y)
        print(self.svm.getSupportVectors())
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

class FeatureMatcher:

    def __init__(self, img_hit_ratio_threshold=0.9, ft_hit_ratio_threshold=0.8):
        self.matcher = cv.FlannBasedMatcher_create()
        self.ft_ext = cv.xfeatures2d.SIFT_create()
        self.label_list = []
        self.img_hit_ratio_threshold = img_hit_ratio_threshold
        self.ft_hit_ratio_threshold= ft_hit_ratio_threshold
        self.keypoints = None

    def fit(self, x, y):
        x_ft = list(map(lambda img: self.ft_ext.detectAndCompute(img, None), x))
        self.keypoints, x_des = list(zip(*x_ft))
        #self.keypoints = list(map(lambda kp: kp.pt, self.keypoints))
        self.matcher.add(x_des)
        test = self.matcher.getTrainDescriptors()
        self.label_list = y

    def predict(self, x):
        x_ft = list(map(lambda img: self.ft_ext.detectAndCompute(img, None), x))
        x_kp, x_des = list(zip(*x_ft))
        matches_per_img = list(map(lambda fts: self.matcher.match(fts), x_des))  
        predictions = self._matches_to_prediction(matches_per_img)
        return np.asarray(predictions)

    def evaluate(self, x, y):
        y = np.asarray(y)
        response = self.predict(x)
        accuracy = _calcute_array_equal_ratio(y, response)
        return accuracy
        
    def _compute_ft(self, x_item):
        kp, ds = self.ft_ext.detectAndCompute(x_item, None)
        return ds

    def _sort_by_imgid(self, matches_from_img):
        matches_by_imgid = {}
        for m_list in matches_from_img:
            for match in m_list:
                if match.imgIdx in matches_by_imgid:
                    matches_by_imgid[match.imgIdx].append(match)
                else:
                    matches_by_imgid[match.imgIdx] = [match]
        matches_sorted = matches_by_imgid.values()
        return matches_sorted
                

    def _matches_to_prediction(self, matches_per_img):
        predictions = []
        for matches in matches_per_img:
            match_labels = list(map(lambda match: self.label_list[match.imgIdx], matches))
            labels_counter = Counter(match_labels)
            most_common = labels_counter.most_common()
            predictions.append(most_common[0][0])
            """if most_common[0][1] - most_common[1][1] > 0:
                predictions.append(most_common[0][0])
            else:
                predictions.append(None)"""
        return predictions


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
            print(self.detect_model.predict([roi]))
            cv.imshow('', roi)
            cv.waitKey(0)
            roi_pos_list.append(win_pos.copy())
            win_pos[0] += self.win_stride[0]
            if win_pos[0] + self.win_size[0] > x.shape[1]:
                win_pos[0] = 0
                win_pos[1] += self.win_stride[1]
        """response = self.detect_model.predict(roi_list)
        true_array = np.full(len(roi_pos_list), true_class)
        equal_array = np.equal(true_array, response)
        roi_array = np.asarray(roi_pos_list)
        shape = roi_array.shape
        roi_pos_array = np.asarray(roi_pos_list)
        roi_array = np.compress(equal_array, roi_pos_list, axis=0)"""
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



