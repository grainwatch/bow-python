from imgclassifier import HOGModel, BOWModel, FeatureMatcher
from objdetector import SlidingWindowDetector
import objdetector

def HOGSlidingWindowDetector(win_size=(128, 128), blocks=(2, 2), cells_per_block=(2, 2), block_stride_in_cells=(1, 1), bins=9, low=0.05, high=0.3, win_stride=(32, 32)) -> SlidingWindowDetector:
    model = HOGModel(win_size, blocks, cells_per_block, block_stride_in_cells, bins)
    detector = SlidingWindowDetector(model, low, high, win_size, win_stride)
    return detector

def BOWSlidingWindowDetector(clusters=150, low=0.05, high=0.3, win_size=(128, 128), win_stride=(32, 32)) -> SlidingWindowDetector:
    model = BOWModel(clusters=clusters)
    detector = SlidingWindowDetector(model, low, high, win_size, win_stride)
    return detector

def FTMatcherSlidingWindowDetector(low=0.05, high=0.3, win_size=(128, 128), win_stride=(32, 32)) -> SlidingWindowDetector:
    model = FeatureMatcher()
    detector = SlidingWindowDetector(model, low, high, win_size, win_stride)
    return detector

ColorDBScan = objdetector.MultiClassColorDBScan