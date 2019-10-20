from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height

    def copy(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    def rescale(self, scale: float) -> Rect:
        x = int(self.x * scale)
        y = int(self.y * scale)
        width = int(self.width * scale)
        height = int(self.height * scale)
        return Rect(x, y, width, height)
    
    def pad(self, padding: Tuple[int, int]) -> Rect:
        x = self.x - padding[0]//2
        y = self.y - padding[1]//2
        width = self.width + padding[0]
        height = self.height + padding[1]
        return Rect(x, y, width, height)

    def get_overlap(self, overlap_rect: Rect) -> int:
        ol_x1 = max(self.x, overlap_rect.x)
        ol_y1 = max(self.y, overlap_rect.y)
        ol_x2 = min(self.x + self.width, overlap_rect.x + overlap_rect.width)
        ol_y2 = min(self.y + self.height, overlap_rect.y + overlap_rect.height)
        ol_width = ol_x2 - ol_x1
        ol_height = ol_y2 - ol_y1
        if ol_width < 0 or ol_height < 0:
            ol_area =  0
        else:
            ol_area = ol_width * ol_height
        return ol_area

    def get_iou(self, overlap_rect: Rect) -> float:
        ol_area = self.get_overlap(overlap_rect)
        union_area = self.area + overlap_rect.area - ol_area
        iou = ol_area / union_area
        return iou

    def find_best_iou_rect(self, rects: List[Rect], threshold: float) -> Rect:
        best_iou = 0
        best_rect = None
        for rect in rects:
            iou = self.get_iou(rect)
            if iou >= threshold and iou > best_iou:
                best_rect = rect
                best_iou = iou
        return best_iou, best_rect

    def get_overlap_with(self, overlap_rect: Rect) -> float:
        ol_area = self.get_overlap(overlap_rect)
        ol = ol_area / self.area
        return ol

    def to_window(self, winsize: Tuple[int, int]) -> Rect:
        win_width, win_height = winsize
        center_x = self.x + self.width/2
        center_y = self.y + self.height/2
        win_x = int(center_x - win_width/2)
        win_y = int(center_y - win_height/2)
        return Rect(win_x, win_y, win_width, win_height)

    def move_to_shape(self, imgshape: Tuple[int, int]) -> Rect:
        img_width, img_height = imgshape[1], imgshape[0]
        moved = self.copy()
        if moved.x < 0:
            moved.x = 0
        elif moved.x + moved.width > img_width:
            moved.x = img_width - moved.width
        if moved.y < 0:
            moved.y = 0
        elif moved.y + moved.height > img_height:
            moved.y = img_height - moved.height
        return moved

    def to_roi(self, img: np.ndarray):
        roi = img[self.y:self.y+self.height, self.x:self.x+self.width]
        return roi
    
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

def filter_overlapping_rects(rects1, rects2, low, high):
    o_rects = []
    for rect1 in rects1:
        for rect2 in rects2:
            iou = rect1.get_iou(rect2)
            if low <= iou <= high:
                o_rects.append(rect2)
    return o_rects

def rects_to_json(rects: Rect, dstpath: Path):
    labeljson_dict = {}
    labels = []
    for rect in rects:
        label_dict = {}
        points = [
            [rect.x, rect.y],
            [rect.x + rect.width, rect.y + rect.height]
        ]
        label_dict['points'] = points
        labels.append(label_dict)
    labeljson_dict['shapes'] = labels
    with dstpath.open('w') as jsonfile:
        json.dump(labeljson_dict, jsonfile, indent='\t')

def rects_labels_to_json(rects: List[Rect], labels: List[int], dstpath: Path):
    labeljson_dict = {}
    shapes = []
    for rect, label in zip(rects, labels):
        shape_dict = {}
        points = [
            [rect.x, rect.y],
            [rect.x + rect.width, rect.y + rect.height]
        ]
        shape_dict['points'] = points
        shape_dict['label'] = label
        shapes.append(shape_dict)
    labeljson_dict['shapes'] = shapes
    with dstpath.open('w') as jsonfile:
        json.dump(labeljson_dict, jsonfile, indent='\t')

def rects_from_json(jsonpath: Path) -> Tuple[List[Rect]]:
    rects = []
    with jsonpath.open() as jsonfile:
        labeljson_dict = json.load(jsonfile)
        shapes = labeljson_dict['shapes']
        for shape in shapes:
            points = shape['points']
            rectangle = _rect_from_points(points)
            rects.append(rectangle)     
    return rects

def rects_labels_from_json(jsonpath: Path) -> Tuple[List[Rect], List[int]]:
    rects = []
    labels = []
    with jsonpath.open() as jsonfile:
        labeljson_dict = json.load(jsonfile)
        shapes = labeljson_dict['shapes']
        for shape in shapes:
            points = shape['points']
            rectangle = _rect_from_points(points)
            rects.append(rectangle)     
            label = shape['label']
            labels.append(label)
    return rects, labels

def _rect_from_points(points) -> Rect:
    x = [0, 0]
    y = [0, 0]
    if points[0][0] < points[1][0]:
        x[0] = points[0][0]
        x[1] = points[1][0]
    else:
        x[0] = points[1][0]
        x[1] = points[0][0]
    if points[0][1] < points[1][1]:
        y[0] = points[0][1]
        y[1] = points[1][1]
    else:
        y[0] = points[1][1]
        y[1] = points[0][1]
    width = x[1] - x[0]
    height = y[1] - y[0]
    return Rect(int(x[0]), int(y[0]), int(width), int(height))

def filter_rects_with_label(rects: List[Rect], labels: List[int], filter_label: int, ignore_half_labels: bool=True, class_amount: int=4) -> Tuple[List[Rect], List[int]]:
    if ignore_half_labels:
        filtered_rects_labels = filter(lambda rect_label: rect_label[1] == filter_label, zip(rects, labels))
    else:
        filtered_rects_labels = filter(lambda rect_label: rect_label[1] == filter_label or rect_label[1] == filter_label + class_amount, zip(rects, labels))
    rects, labels = list(zip(*filtered_rects_labels))
    return rects