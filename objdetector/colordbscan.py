from typing import List, Tuple
from math import hypot
import util.rectangle as rectangle
import numpy as np
import cv2 as cv
import sklearn.metrics
import sklearn.cluster
from objdetector.objectdetector import ObjectDetector

Boundary = Tuple[float, float, float]


class ColorDBScan(ObjectDetector):
    def __init__(self, label: int, lower_bounds: Boundary=(0, 150, 150), upper_bounds: Boundary=(50, 255, 255), minpts: int=40, eps: float=10):
        self.label = label
        self.lower_bounds = lower_bounds
        self.higher_bounds = upper_bounds
        self.minpts = minpts
        self.eps = eps

    def _predict_img(self, img):
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        self.inrange_img = cv.inRange(hsv_img, self.lower_bounds, self.higher_bounds)
        pts = cv.findNonZero(self.inrange_img)
        if pts is None:
            return [], []
        else:
            pts = np.reshape(pts, (pts.shape[0], pts.shape[2]))
            rects_sk = self.sk_dbscan(pts)
            labels = [self.label] * len(rects_sk)
            #rects = self.dbscan(pts.tolist())
            return rects_sk, labels
        
    def sk_dbscan(self, pts):
        dbscan = sklearn.cluster.DBSCAN(self.eps, self.minpts)
        labels = dbscan.fit_predict(pts)
        rects = []
        for label in set(labels):
            if label == -1:
                continue
            idx = np.where(labels == label)
            cluster_pts = np.take(pts, idx, 0)[0]
            rect = self._rect_from_border_pts(cluster_pts.tolist())
            rects.append(rect)
        return rects

    def dbscan(self, pts):
        rects = []
        while len(pts) > 0:
            core_pt = pts.pop()
            border_pts = []
            nb_pts, indexes = self._find_neighbour(core_pt, pts, self.eps)
            if len(nb_pts) >= self.minpts:
                self.remove_pts(pts, indexes)
                query_nb_pts = nb_pts.copy()
                while len(query_nb_pts) > 0:
                    nb_pt = query_nb_pts.pop()
                    nb_pts, indexes = self._find_neighbour(nb_pt, pts, self.eps)
                    if len(nb_pts) >= self.minpts:
                        self.remove_pts(pts, indexes)
                        query_nb_pts += nb_pts
                    else:
                        border_pts.append(nb_pt)
                rect = self._rect_from_border_pts(border_pts)
                rects.append(rect)
        return rects

    def remove_pts(self, pts, indexes):
        indexes = sorted(indexes, reverse=True)
        for i in indexes:
            del pts[i]
    
    def _find_neighbour(self, refpt, pts, radius):
        radius_pts = []
        indexes = []
        for i, pt in enumerate(pts):
            distance = self.get_distance(refpt, pt)
            if distance <= radius:
                radius_pts.append(pt)
                indexes.append(i)
        return radius_pts, indexes

    def get_distance(self, pt1, pt2):
        diff_pt_0 = abs(pt1[0] - pt2[0])
        diff_pt_1 = abs(pt1[1] - pt2[1])
        return hypot(diff_pt_0, diff_pt_1)

    def _rect_from_border_pts(self, border_pts):
        pt = border_pts.pop()
        l_pt = pt[0]
        r_pt = pt[0]
        u_pt = pt[1]
        b_pt = pt[1]
        for pt in border_pts:
            if pt[0] < l_pt:
                l_pt = pt[0]
            elif pt[0] > r_pt:
                r_pt = pt[0]
            if pt[1] < u_pt:
                u_pt = pt[1]
            elif pt[1] > b_pt:
                b_pt = pt[1]
        width = r_pt - l_pt
        height = b_pt - u_pt
        return rectangle.Rect(l_pt, u_pt, width, height)

Parameters = List[Tuple[int, Boundary, Boundary, int, float]]
class MultiClassColorDBScan(ObjectDetector):
    def __init__(self, detector_params: Parameters):
        self.detectors = []
        for params in detector_params:
            label, lower_bounds, upper_bounds, minpts, eps = params
            detector = ColorDBScan(label, lower_bounds, upper_bounds, minpts, eps)
            self.detectors.append(detector)

    def _predict_img(self, img: np.ndarray):
        rects, labels = [], []
        for detector in self.detectors:
            class_rects, class_labels = detector._predict_img(img)
            rects += class_rects
            labels += class_labels
        return rects, labels