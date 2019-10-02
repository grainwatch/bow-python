import unittest
import model
import cv2 as cv

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


if __name__ == '__main__':
    unittest.main()