import unittest
from pathlib import Path
import objdataset
from rectangle import Rect
import cv2 as cv
import numpy as np

class TestObjDataset(unittest.TestCase):
    def test_rect_from_points_1(self):
        points = ((123, 23), (420, 100))
        result = objdataset._rect_from_points(points)
        expected = Rect(123, 23, 297, 77)
        self.assertEqual(expected, result)

    def test_rect_from_points_2(self):
        points = ((740, 211), (567, 15))
        result = objdataset._rect_from_points(points)
        expected = Rect(567, 15, 173, 196)
        self.assertEqual(expected, result)

    def test_rects_from_json(self):
        jsonpath = Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\test.json')
        result = objdataset._rects_from_json(jsonpath)
        expected = [
            Rect(50, 70, 50, 70),
            Rect(40, 60, 110, 40)
        ]
        self.assertListEqual(expected, result)

    def test_load_labels(self):
        labelfolder = Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\objdataset')
        result_dict = objdataset._load_labels(labelfolder)
        expected_dict = {'testclass': {'test': [Rect(50, 70, 50, 70), Rect(40, 60, 110, 40)]}}
        self.assertDictEqual(expected_dict, result_dict)

    def test_load_img_dict(self):
        test_img = cv.imread('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\objdataset\\testclass\\test.jpg')
        result_dict = objdataset._load_img_dict(Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\objdataset'))
        expected_dict = {'testclass': {'test': test_img}}
        self.assertListEqual(list(expected_dict.keys()), list(result_dict.keys()))
        resultclass_dict = result_dict['testclass']
        expectedclass_dict = expected_dict['testclass']
        self.assertListEqual(list(expectedclass_dict.keys()), list(resultclass_dict.keys()))
        self.assertTrue(np.array_equal(expectedclass_dict['test'], resultclass_dict['test']))
        
    def test_flatten_img_dict(self):
        imgs = [np.random.randint(0, 255, (100, 100, 3), np.uint8) for i in range(3)]
        rects = [
            [Rect(0, 0, 100, 100), Rect(50, 70, 200, 300), Rect(10, 5, 10, 5)],
            [Rect(14, 56, 145, 14), Rect(10, 20, 50, 50), Rect(0, 0, 400, 300)],
            [Rect(20, 10, 300, 400), Rect(50, 50, 50, 50)]
        ]
        labels = [1 for i in range(3)]
        imgclass_dict = dict([(str(i), imgs[i]) for i in range(3)])
        rectclass_dict = dict([(str(i), rects[i]) for i in range(3)])
        img_dict = {'corn': imgclass_dict}
        rect_dict = {'corn': rectclass_dict}
        result_imgs, result_rects, result_labels = objdataset._flatten_img_dict(img_dict, rect_dict)
        self.assertListEqual(imgs, result_imgs)
        self.assertListEqual(rects, result_rects)
        self.assertListEqual(labels, result_labels)

    def test_load_grain_dataset(self):
        imgfolder = 'C:\\Users\\Alex\\Desktop\\imgdataset'
        labelfolder = 'C:\\Users\\Alex\\IdeaProjects\\grain-swpt\\dataset'
        (train_imgs, train_rects, train_labels), (test_imgs, test_rects, test_labels) = objdataset.load_grain_dataset(imgfolder, labelfolder)
        pass

if __name__ == '__main__':
    unittest.main()