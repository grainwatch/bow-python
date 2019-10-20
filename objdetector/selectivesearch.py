import numpy as np
import cv2 as cv
from objdetector.objectdetector import ObjectDetector
import util.rectangle as rectangle

class SelectiveSearchDetector(ObjectDetector):
    def __init__(self, model, low, high):
        super().__init__(model, low, high)
        self.sel_search = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    def _search_rects(self, img):
        self.sel_search.setBaseImage(img)
        self.sel_search.switchToSingleStrategy()
        sel_rects = self.sel_search.process()
        sel_rects = list(map(lambda rect: rectangle.Rect(rect[0], rect[1], rect[2], rect[3]), sel_rects))
        return sel_rects
