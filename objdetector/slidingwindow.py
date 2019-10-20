import numpy as np
import cv2 as cv

from util import rectangle
from objdetector.objectdetector import LearningObjectDetector

class SlidingWindowDetector(LearningObjectDetector):
    def __init__(self, model, low=.05, high=.3, winsize=(128, 128), winstride=(32,32)):
        super().__init__(model, low, high)
        self.winsize = winsize
        self.winstride = winstride

    def _search_rects(self, img):
        x_pos = 0
        y_pos = 0
        rects = []
        while y_pos + self.winsize[1] < img.shape[0]:
            rect = rectangle.Rect(x_pos, y_pos, self.winsize[0], self.winsize[1])
            rects.append(rect)
            x_pos += self.winstride[0]
            if x_pos + self.winsize[0] > img.shape[1]:
                x_pos = 0
                y_pos += self.winstride[1]
        return rects

    def _create_truth_rois(self, img, labeled_rects):
        win_rects = map(lambda rect: rect.to_window((128, 128)), labeled_rects)
        truth_rects = map(lambda rect: rect.move_to_shape(img.shape), win_rects)
        truth_rois = list(map(lambda rect: rect.to_roi(img), truth_rects))
        return truth_rois