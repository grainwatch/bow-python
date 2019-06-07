import cv2 as cv
import numpy as np
import json
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

img_folder = 'C:\\Users\\Alex\\IdeaProjects\\grain-swpt\\dataset\\corn\\'

def main():
    feature_extractor = cv.AKAZE_create(cv.AKAZE_DESCRIPTOR_KAZE)
    img_paths = list(Path(img_folder).glob('*.JPG'))
    part_img_paths = img_paths[:4]
    trainer = BOWTrainer(feature_extractor, clusters=20, threads=4)
    voc = trainer.train(part_img_paths)
    bow_extractor = BOWDescriptor(feature_extractor, voc)
    histo_list = None
    for path in img_paths[:4]:
        img = cv.imread(str(path))
        img = cv.pyrDown(img)
        rect_points = get_rect_points(str(path))
        for rect in rect_points:
            histo = bow_extractor.detect_roi(img, rect)
            print(rect)
            print(histo)
            if histo_list is None:
                histo_list = histo
            else:
                histo_list = np.append(histo_list, histo, axis=0)
    #histo_list = histo_list.reshape((histo_list.shape[0], histo_list.shape[1], 1))
    label_list = np.asarray([1]*histo_list.shape[0], dtype=np.int32)
    #histo_list = histo_list.transpose()
    print(label_list)
    print(histo_list)
    print(histo_list.shape)
    label_list = label_list.reshape((72, 1))
    #label_list = cv.UMat(label_list)
    classifier = BOWClassifier()
    classifier.train(histo_list, label_list)



def test_roi(img, points, ft):
    top_left, bottom_right = get_rect(points)
    roi_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    kp, ds = ft.detectAndCompute(roi_img, None)
    return ds


def detect_roi(img, mask, points, ft):
    top_left, bottom_right = get_rect(points)
    mask = cv.rectangle(mask, top_left, bottom_right, 255, cv.FILLED)
    kp, ds = ft.detectAndCompute(img, mask)
    mask.fill(0)
    print(f'Found {len(kp)} in img')
    return kp, ds


def get_rect_points(img_filename: str):
    index = img_filename.find('.')
    json_filename = img_filename[:index] + '.json'
    json_fp = open(json_filename)
    json_object = json.load(json_fp)
    shapes = json_object['shapes']
    return [shape['points'] for shape in shapes]


class BOWTrainer:
    INITIAL_TASKS = 8
    PYR_STEPS = 2

    def __init__(self, ft_extractor, clusters=100, threads=8):
        if threads < 8:
            self.initial_tasks = threads
        else:
            self.initial_tasks = BOWTrainer.INITIAL_TASKS
        self.ft_extractor = ft_extractor
        self.worker = ThreadPoolExecutor(max_workers=threads)
        self.img_lock = Lock()
        self.img_paths = []
        self.bow_trainer = cv.BOWKMeansTrainer(clusters)
        self.bow_lock = Lock()
        self.finish_counter = 0

    def train(self, img_paths: List):
        print(img_paths)
        self.img_paths = img_paths
        print(self.img_paths)
        for x in range(self.initial_tasks):
            self._create_quest()
        while self.finish_counter < self.initial_tasks:
            sleep(1)
        return self.bow_trainer.cluster()

    def on_img_trained(self, result):
        result.result()
        with self.img_lock:
            if len(self.img_paths) > 0:
                self._create_quest()
            else:
                self.finish_counter += 1

    def _train_img(self, path):
        points_list = get_rect_points(str(path))
        img = cv.imread(str(path))
        img = cv.pyrDown(img)
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        for points in points_list:
            kp, ds = detect_roi(img, mask, points, self.ft_extractor)
            with self.bow_lock:
                self.bow_trainer.add(ds)

    def _create_quest(self):
        print(self.img_paths)
        path = self.img_paths.pop(0)
        fut = self.worker.submit(self._train_img, path)
        fut.add_done_callback(self.on_img_trained)


class BOWDescriptor:
    def __init__(self, ft_extractor, voc):
        self.matcher = cv.FlannBasedMatcher_create()
        self.matcher.add(voc)
        self.ft_extractor = ft_extractor
        self.clusters = len(voc)
        self.mask = None

    def compute(self, query_ds):
        matches = self.matcher.match(query_ds)
        histogramm = np.zeros((1, self.clusters), dtype=np.float32)
        for x in matches:
            print(len(self.matcher.getTrainDescriptors()))
            print(f'distance: {x.distance} imgid: {x.imgIdx} queryid: {x.queryIdx} trainid: {x.trainIdx}')
            i = x.imgIdx
            histogramm[0][i] += 1
        return histogramm

    def detect_roi(self, img, roi):
        if self.mask is None:
            self.mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        top_left, bottum_right = get_rect(roi)
        cv.rectangle(self.mask, top_left, bottum_right, 255, cv.FILLED)
        kp, ds = self.ft_extractor.detectAndCompute(img, self.mask)
        self.mask.fill(0)
        return self.compute(ds)


class BOWClassifier:
    def __init__(self):
        self.svm = cv.ml.SVM_create()

    def train(self, histo_list, labels):
        print(labels)
        print(labels.shape)
        self.svm.trainAuto(histo_list, cv.ml.ROW_SAMPLE, labels)
        self.svm.save('svm')
        print('fertig')

    def predict(self, histo):
        return self.svm.predict(histo)


def get_rect(points):
    p1 = points[0]
    p2 = points[1]
    x1 = round(p1[0] * 0.5)
    y1 = round(p1[1] * 0.5)
    x2 = round(p2[0] * 0.5)
    y2 = round(p2[1] * 0.5)
    top_left = [0, 0]
    bottom_right = [0, 0]
    if x1 < x2:
        top_left[0] = x1
        bottom_right[0] = x2
    else:
        top_left[0] = x2
        bottom_right[0] = x1
    if y1 < y2:
        top_left[1] = y1
        bottom_right[1] = y2
    else:
        top_left[1] = y2
        bottom_right[1] = y1
    return tuple(top_left), tuple(bottom_right)


if __name__ == '__main__':
    main()