import unittest
from pathlib import Path
import rectangle

class TestRect(unittest.TestCase):
    def test_dataclass(self):
        rect = rectangle.Rect(0, 0, 100, 200)

    def test_get_area(self):
        rect = rectangle.Rect(12, 43, 100, 200)
        result = rect.area
        expected = 20000
        self.assertEqual(expected, result)

    def test_get_iou(self):
        rect1 = rectangle.Rect(0, 0, 100, 200)
        rect2 = rectangle.Rect(50, 100, 100, 200)
        result = rectangle.get_iou(rect1, rect2)
        expected = 5000 / 35000
        self.assertEqual(expected, result)

    def test_in_iou_range_1(self):
        rect1 = rectangle.Rect(0, 0, 100, 200)
        rect2 = rectangle.Rect(50, 100, 100, 200)
        result = rectangle.in_iou_range(rect1, rect2, 0.2, 0.5)
        self.assertFalse(result)
    
    def test_in_iou_range_2(self):
        rect1 = rectangle.Rect(0, 0, 100, 200)
        rect2 = rectangle.Rect(50, 100, 100, 200)
        result = rectangle.in_iou_range(rect1, rect2, 0.1, 0.3)
        self.assertTrue(result)

    def test_in_iou_range_3(self):
        rect1 = rectangle.Rect(0, 0, 100, 200)
        rect2 = rectangle.Rect(50, 100, 100, 200)
        result = rectangle.in_iou_range(rect1, rect2, 0.0, 0.1)
        self.assertFalse(result)

    def test_get_overlap_with_1(self):
        rect1 = rectangle.Rect(100, 100, 100, 100)
        rect2 = rectangle.Rect(50, 170, 200, 100)
        result = rect1.get_overlap_with(rect2)
        expected = 3/10
        self.assertEqual(expected, result)

    def test_get_overlap_with_2(self):
        rect1 = rectangle.Rect(100, 100, 100, 100)
        rect2 = rectangle.Rect(120, 120, 60, 60)
        result = rect1.get_overlap_with(rect2)
        expected = 36/100
        self.assertEqual(expected, result)

    def test_get_difficult_negatives_rects(self):
        rects1 = [
            rectangle.Rect(100, 100, 100, 100)
        ]
        rects2 = [
            rectangle.Rect(200, 200, 100, 100),
            rectangle.Rect(50, 160, 200, 100)
        ]
        result = rectangle.filter_overlapping_rects(rects1, rects2, 0.3, 0.5)
        expected = [rects2[1]]
        self.assertListEqual(expected, result)

    def test_rescale(self):
        rect = rectangle.Rect(20, 40, 100, 200)
        result = rect.rescale(0.5)
        expected = rectangle.Rect(10, 20, 50, 100)
        self.assertEqual(expected, result)

    def test_from_to_json(self):
        rects = [
            rectangle.Rect(10, 24, 100, 245),
            rectangle.Rect(12, 10, 500, 200)
        ]
        testpath = Path('test\\rectangle\\test.json')
        rectangle.rects_to_json(rects, testpath)
        result = rectangle.rects_from_json(testpath)
        self.assertListEqual(rects, result)


if __name__ == '__main__':
    unittest.main()