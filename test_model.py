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
        self.detector = model.SelectiveSearchDetector(bowmodel, 0.3, 0.5)
        self.img = np.random.randint(0, 256, (1000, 500, 3))
        self.rects = [
            Rect(100, 100, 50, 100),
            Rect(400, 200, 200, 100),
            Rect(700, 300, 150, 150)
        ]
        self.labels = [
            1,
            1,
            1
        ]

    def test_get_ground_truth_rois(self):
        result_rois, result_labels = self.detector._get_ground_truth_rois([self.img], [self.rects], [self.labels])
        expected_rois = [
            self.img[100:100+50, 100:100+50],
            self.img[200:200+100, 400:400+200],
            self.img[300:300+150, 700:700+150]
        ]
        expected_labels = [
            1,
            1,
            1
        ]
        self.assertListEqual(expected_rois, result_rois)
        self.assertListEqual(expected_labels, result_labels)


if __name__ == '__main__':
    unittest.main()