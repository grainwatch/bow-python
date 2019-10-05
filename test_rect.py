import unittest
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

if __name__ == '__main__':
    unittest.main()