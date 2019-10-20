import cv2 as cv
import numpy as np

class BOWModel:
    def __init__(self, clusters=150):
        self.clusters = clusters
        self.svm = cv.ml.SVM_create()
        self.bow_trainer = cv.BOWKMeansTrainer(clusters)
        self.matcher = cv.BFMatcher_create()
        self.ft_ext = cv.KAZE_create()
        #self.ft_ext = cv.xfeatures2d.SIFT_create()
        self.x_ft = None
        self.x_histogram = None
        self.vocabulary = None
        self.fitted = False

    def fit(self, x, y):
        if self.fitted:
            self.bow_trainer.clear()
            self.matcher.clear()
        self.x = x
        self.x_ft = map(self._compute_ft, x)
        x_y = zip(self.x_ft, y)
        filterd_x_y = filter(lambda ft_label: ft_label[0] is not None, x_y)
        self.x_ft, y = list(zip(*filterd_x_y))
        for ft in self.x_ft:
            self.bow_trainer.add(ft)
        self.vocabulary = self.bow_trainer.cluster()
        self.matcher.add([self.vocabulary])
        self.x_histogram = list(map(self._compute_histogram, self.x_ft))
        self.x_histogram = np.asarray(self.x_histogram, np.float32)
        y_array = np.asarray(y, np.int32)
        print('training svm')
        #self.svm.train(self.x_histogram, cv.ml.ROW_SAMPLE, y_array)
        self.svm.trainAuto(self.x_histogram, cv.ml.ROW_SAMPLE, y_array)
        self.fitted = True

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
        if self.fitted:
            x_ft = map(self._compute_ft, x)
            x_histogram = map(self._compute_histogram, x_ft)
            x_histogram = np.asarray(list(x_histogram), np.float32)
            response = self.svm.predict(x_histogram)[1]
            response = response.astype(np.int).reshape(response.shape[0])
            return response
    
    def evaluate(self, x, y):
        y = np.asarray(y)
        response = self.predict(x)
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