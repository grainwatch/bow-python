import unittest
import dataset
from pathlib import Path
import cv2 as cv
import numpy as np

class TestDataset(unittest.TestCase):
    #TODO: more test cases

    def test_scale_rectangle(self):
        roi = (10, 20, 100, 120)
        result = dataset._scale_rectangle(roi, 0.5)
        expected = (5, 10, 50, 60)
        self.assertTupleEqual(expected, result)

    def test_rectangle_to_window(self):
        roi = (30, 40, 100, 120)
        result = dataset._rectangle_to_window(roi, (128, 128))
        expected = (16, 36, 128, 128)
        self.assertTupleEqual(expected, result)

    def test_rectangle_from_points_1(self):
        points = ((123, 23), (420, 100))
        result = dataset._rectangle_from_points(points)
        expected = (123, 23, 297, 77)
        self.assertEqual(expected, result)

    def test_rectangle_from_points_2(self):
        points = ((740, 211), (567, 15))
        result = dataset._rectangle_from_points(points)
        expected = (567, 15, 173, 196)
        self.assertEqual(expected, result)

    def test_is_roi_colliding_false_1(self):
        roi_to_check = (50, 100, 100, 100)
        rois = [
            (0, 0, 50, 50),
            (160, 250, 100, 100)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertFalse(result)

    def test_is_roi_colliding_false_2(self):
        roi_to_check = (5, 5, 100, 100)
        rois = [
            (105, 105, 100, 100)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertFalse(result)

    def test_is_roi_colliding_true_1(self):
        roi_to_check = (50, 50, 100, 100)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)
    
    def test_is_roi_colliding_true_2(self):
        roi_to_check = (250, 50, 100, 100)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_3(self):
        roi_to_check = (50, 250, 100, 100)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_4(self):
        roi_to_check = (250, 250, 100, 100)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_5(self):
        roi_to_check = (50, 50, 350, 350)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_6(self):
        roi_to_check = (50, 50, 350, 150)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_7(self):
        roi_to_check = (50, 200, 350, 150)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_8(self):
        roi_to_check = (50, 50, 150, 350)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_9(self):
        roi_to_check = (200, 50, 150, 350)
        rois = [
            (100, 100, 200, 200)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_is_roi_colliding_true_10(self):
        roi_to_check = (5, 5, 100, 100)
        rois = [
            (5, 15, 100, 100)
        ]
        result = dataset.is_roi_colliding(roi_to_check, rois)
        self.assertTrue(result)

    def test_rois_from_json(self):
        jsonpath = Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\test.json')
        result = dataset._rois_from_json(jsonpath)
        expected = [
            (50, 70, 50, 70),
            (40, 60, 110, 40)
        ]
        self.assertListEqual(expected, result)
    
    def test_split_samples(self):
        samples =  [x for x in range(11)]
        test_amount = 0.3
        result_trainsamples, result_testsamples = dataset._split_samples(samples, test_amount)
        expected_trainsamples = samples[:7]
        expected_testsamples = samples[7:]
        self.assertListEqual(expected_trainsamples, result_trainsamples)
        self.assertListEqual(expected_testsamples, result_testsamples)

    def test_samples_from_img(self):
        img = (np.random.rand(200, 200) * 255).astype(np.uint8)
        rois = [
            (50, 50, 100, 100),
            (100, 100, 50, 50)
        ]
        result = dataset._samples_from_img(img, rois)
        expected = [
            img[50:150, 50:150],
            img[100:150, 100:150]
        ]
        self.assertEqual(len(expected), len(result))
        self.assertTrue(np.array_equal(expected[0], result[0]))
        self.assertTrue(np.array_equal(expected[1], result[1]))

    def test_name_from_path(self):
        path = Path('test.jpg')
        result = dataset._name_from_path(path)
        expected = 'test'
        self.assertEqual(expected, result)

    def test_move_into_img_1(self):
        roi = (-5, -10, 128, 128)
        img_shape = (1000, 500)
        result = dataset._move_window_into_img(roi, img_shape)
        expected = (0, 0, 128, 128)
        self.assertTupleEqual(expected, result)

    def test_move_into_img_2(self):
        roi = (1000, 500, 128, 128)
        img_shape = (500, 1000)
        result = dataset._move_window_into_img(roi, img_shape)
        expected = (872, 372, 128, 128)
        self.assertTupleEqual(expected, result)

    def test_negative_samples_from_imgs(self):
        jsonpath = Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\DSC_0922.json')
        rois = dataset._rois_from_json(jsonpath)
        scaled_rois = list(map(lambda roi: dataset._scale_rectangle(roi, 0.3), rois))
        img = cv.imread('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\DSC_0922.jpg')
        Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\negative').mkdir(exist_ok=True)
        img = cv.resize(img, (0, 0), None, 0.3, 0.3)
        imgs = dataset._negative_samples_from_img(img, scaled_rois, (128, 128), (128, 128))
        for i in range(len(imgs)):
            cv.imwrite(f'C:\\Users\\Alex\\CodeProjects\\bow-python\\test\\dataset\\negative\\{i}.jpg', imgs[i])

if __name__ == '__main__':
    unittest.main()
