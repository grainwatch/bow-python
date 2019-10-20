from collections import Counter
import numpy as np
import cv2 as cv
import sklearn.metrics as metrics

class FeatureMatcher:
    def __init__(self):
        self.matcher = cv.FlannBasedMatcher_create()
        self.ft_ext = cv.xfeatures2d.SIFT_create()
        self.label_list = []
        self.keypoints = None

    def fit(self, x, y):
        x_ft = list(map(lambda img: self.ft_ext.detectAndCompute(img, None), x))
        self.keypoints, x_des = list(zip(*x_ft))
        self.matcher.add(x_des)
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
        accuracy = metrics.accuracy_score(y, response)
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
            if len(match_labels) == 0:
                predictions.append(-1)
            else:
                labels_counter = Counter(match_labels)
                most_common = labels_counter.most_common()
                predictions.append(most_common[0][0])
        return predictions