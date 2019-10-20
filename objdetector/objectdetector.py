import abc
from pathlib import Path
from typing import List, Tuple
from collections import Counter
import numpy as np
import cv2 as cv
import sklearn.metrics
import util.rectangle as rectangle
import dataset
import random

IOU_DECISION = 0.5

DEBUG_PATH = Path('C:\\Users\\Alex\\Desktop\\debug')

CLASSES = [1, 2, 3, 4]

PADDING = (8, 8)

class ObjectDetector(abc.ABC):
    def predict(self, imgs: List[np.ndarray]) -> Tuple[List[List[rectangle.Rect]], List[List[int]]]:
        img_predicted_rects = []
        img_predicted_labels = []
        for img in imgs:
            predicted_rects, predicted_labels = self._predict_img(img)
            img_predicted_rects.append(predicted_rects)
            img_predicted_labels.append(predicted_labels)
        return img_predicted_rects, img_predicted_labels

    @abc.abstractmethod
    def _predict_img(self, img):
        pass

    def evaluate(self, imgs: List[np.ndarray], img_truth_rects: List[List[rectangle.Rect]], img_truth_labels: List[List[int]], ignore_train_half_labels: bool=True) -> Tuple[float, float]:
        img_predicted_rects, img_predicted_labels = self.predict(imgs)
        all_result_true, all_result_predicted = get_imgs_classification_result(img_truth_rects, img_truth_labels, img_predicted_rects, img_predicted_labels, ignore_train_half_labels)
        aps = []
        for corn_class in CLASSES:
            results = zip(all_result_true, all_result_predicted)
            results = list(filter(lambda result: (result[0] == corn_class and result[1] == corn_class) or (result[0] == -1 and result[1] == corn_class) or (result[0] == corn_class and result[1] == -1), results))
            if len(results) != 0:
                class_result_true, class_result_predicted = list(zip(*results))
                ap = sklearn.metrics.average_precision_score(np.asarray(class_result_true, np.int32), np.asarray(class_result_predicted, np.int32), pos_label=corn_class)
                aps.append(ap)
        m_ap_sum = 0
        for ap in aps:
            m_ap_sum += ap
        m_ap = m_ap_sum / len(aps)
        confusion_matrix = sklearn.metrics.confusion_matrix(all_result_true, all_result_predicted, labels=[-1, 1])
        return m_ap, aps, confusion_matrix

class LearningObjectDetector(ObjectDetector):
    @abc.abstractmethod
    def __init__(self, model, low, high):
        self.model = model
        self.low = low
        self.high = high

    def fit(self, imgs, img_rects, img_labels, img_searched_rects=None):
        all_truth_rois, all_truth_labels, all_negative_rois, all_negative_labels = [], [], [], []
        i = 0
        ndebug = DEBUG_PATH.joinpath('nexamp')
        ndebug.mkdir(exist_ok=True, parents=True)
        pdebug = DEBUG_PATH.joinpath('pexamp')
        pdebug.mkdir(exist_ok=True, parents=True)
        if img_searched_rects is None:
            img_searched_rects = []
            for img, labeled_rects, labels in zip(imgs, img_rects, img_labels):
                truth_rois, truth_labels, negative_rois, negative_labels, searched_rects = self._search_rois(img, labeled_rects, labels)
                all_truth_rois += truth_rois
                all_truth_labels += labels
                all_negative_rois += negative_rois
                all_negative_labels += negative_labels
                img_searched_rects.append(searched_rects)
        else:
            for img, labeled_rects, labels, searched_rects in zip(imgs, img_rects, img_labels, img_searched_rects):
                truth_rois, truth_labels, negative_rois, negative_labels, searched_rects =  self._search_rois(img, labeled_rects, labels, searched_rects)
                all_truth_rois += truth_rois
                all_truth_labels += truth_labels
                all_negative_rois += negative_rois
                all_negative_labels += negative_labels
        print('learning')
        negative_samples = list(zip(all_negative_rois, all_negative_labels))
        negative_samples = random.sample(negative_samples, 8000)
        all_negative_rois, all_negative_labels = zip(*negative_samples)
        for roi in all_negative_rois:
            name = ndebug.joinpath(f'{i}.jpg')
            cv.imwrite(str(name), roi)
            i += 1
        i = 0
        for roi in all_truth_rois:
            name = pdebug.joinpath(f'{i}.jpg')
            cv.imwrite(str(name), roi)
            i += 1
        rois, labels = dataset.shuffle(all_truth_rois + list(all_negative_rois), all_truth_labels + list(all_negative_labels))
        self.model.fit(rois, labels)

    def _predict_img(self, img):
        searched_rects = self._search_rects(img)
        rois = list(map(lambda rect: rect.to_roi(img), searched_rects))
        labels = self.model.predict(rois)
        response = zip(searched_rects, labels)
        response = list(filter(lambda response: response[1] != -1, response))
        if len(response) == 0:
            return [], []
        else:
            filtered_rects, filtered_labels = list(zip(*response))
        return filtered_rects, filtered_labels

    def _search_rois(self, img, labeled_rects, labels, searched_rects=None):
        truth_rois = self._create_truth_rois(img, labeled_rects)
        negative_rois, negative_labels, searched_rects = self._create_negative_samples(img, labeled_rects, searched_rects)
        return truth_rois, labels, negative_rois, negative_labels, searched_rects
        
    def _create_truth_rois(self, img, labeled_rects):
        padded_rects = map(lambda rect: rect.pad(PADDING), labeled_rects)
        padded_rects = map(lambda rect: rect.move_to_shape(img.shape), padded_rects)
        truth_rois = list(map(lambda rect: rect.to_roi(img), padded_rects))
        return truth_rois

    def _create_negative_samples(self, img, labeled_rects, searched_rects=None):
        if searched_rects is None:
            searched_rects = self._search_rects(img)
        negative_rects = rectangle.filter_overlapping_rects(labeled_rects, searched_rects, self.low, self.high)
        negative_rois = list(map(lambda rect: rect.to_roi(img), negative_rects))
        negative_labels = [-1] * len(negative_rois)
        return negative_rois, negative_labels, searched_rects

    @abc.abstractmethod
    def _search_rects(self, img):
        pass


def get_imgs_classification_result(img_truth_rects, img_truth_labels, img_predicted_rects, img_predicted_labels, ignore_train_half_labels: bool=True):
    all_result_predicted = []
    all_result_true = []
    for truth_rects, truth_labels, predicted_rects, predicted_labels in zip(img_truth_rects, img_truth_labels, img_predicted_rects, img_predicted_labels):
        result_true, result_predicted = get_classification_result(truth_rects, truth_labels, predicted_rects, predicted_labels, ignore_train_half_labels)
        all_result_true += result_true
        all_result_predicted += result_predicted
    return all_result_true, all_result_predicted

DIFFICULT_LABELS = [5, 6, 7, 8]

def get_classification_result(truth_rects, truth_labels, predicted_rects, predicted_labels, ignore_train_half_labels: bool=True):
    result_predicted = []
    result_true = []
    truth_label_set = set(truth_labels)
    predicted_label_set = set(predicted_labels)
    for label in truth_label_set & predicted_label_set:
        class_truth_rects = rectangle.filter_rects_with_label(truth_rects, truth_labels, label, ignore_train_half_labels)
        class_predicted_rects = rectangle.filter_rects_with_label(predicted_rects, truth_labels, label)
        detected_truth_rects = []
        for predicted_rect in class_predicted_rects:
            iou, truth_rect = predicted_rect.find_best_iou_rect(class_truth_rects, IOU_DECISION)
            if truth_rect in detected_truth_rects:
                continue #skip it
            else:
                detected_truth_rects.append(truth_rect)
            if truth_rect is None: #false positive
                result_predicted.append(int(label))
                result_true.append(-1)
            else: #true positive
                result_predicted.append(int(label))
                result_true.append(int(label))
        for i in range(len(class_truth_rects) - len(detected_truth_rects)): #false negative
            result_predicted.append(-1)
            result_true.append(int(label))
    predicted_label_counts = Counter(predicted_labels)
    truth_label_counts = Counter(truth_labels)
    for false_positve_label in predicted_label_set - truth_label_set:
        result_predicted += [int(false_positve_label)] * predicted_label_counts[false_positve_label]
        result_true += [1] * predicted_label_counts[false_positve_label]
    for false_negative_label in truth_label_set - predicted_label_set:
        if ignore_train_half_labels:
            if false_negative_label in DIFFICULT_LABELS:
                continue
        result_predicted += [-1] * truth_label_counts[false_negative_label]
        result_true += [int(false_negative_label)] * truth_label_counts[false_negative_label]
    return result_true, result_predicted