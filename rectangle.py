from typing import Tuple
from dataclasses import dataclass

@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height

def get_iou(rect1: Rect, rect2: Rect) -> float:
    diff_width = abs(rect1.x - rect2.x)
    diff_height = abs(rect1.y - rect2.y)
    iou_width = 0
    iou_height = 0
    if rect1.x > rect2.x:
        iou_width = rect1.width - diff_width
    else:
        iou_width = rect2.width - diff_width
    if rect1.y > rect2.y:
        iou_height = rect1.height - diff_height
    else:
        iou_height = rect2.height - diff_height
    area_over = iou_width * iou_height
    area_union = rect1.area + rect2.area - area_over
    iou = area_over / area_union
    return iou

def in_iou_range(rect1: Rect, rect2: Rect, low: float, high):
    iou = get_iou(rect1, rect2)
    if iou < low:
        return False
    if iou > high:
        return False
    return True