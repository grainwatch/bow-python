import unittest
import model
import cv2 as cv
import numpy as np
from rectangle import Rect

class TestObjectDetector(unittest.TestCase):
    def testDetect(self):
        hog = model.HOG.load('hog', 'hogsvm')
        obj_dt = model.ObjectDetector(hog, (128, 128), (32, 32))
        test_img = cv.imread('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\DSC_0922.JPG')
        test_img = cv.resize(test_img, (0, 0), None, 0.3, 0.3)
        rois = obj_dt.detect(test_img)
        for roi in rois:
            test_img = cv.rectangle(test_img, (roi[0], roi[1], 128, 128), (0, 255, 0))
        test_img = cv.resize(test_img, (0, 0), None, 0.5, 0.5)
        cv.imshow('', test_img)
        cv.waitKey(0)

class TestSelectiveSearchDetector(unittest.TestCase):
    def setUp(self):
        bowmodel = model.BOWModel()
        self.detector = model.SelectiveSearchDetector(model, 0.3, 0.5)
        self.imgs = [np.random.randint(0, 256, (1000, 500, 3)) for i in range(5)]
        self.rects = [
            Rect(100, 100, 50, 100),
            Rect(400, 200, 200, 100),
            Rect(700, 300, 150, 150)
        ]

    def test_get_ground_truth_rois(self):
        self.


if __name__ == '__main__':
    unittest.main()