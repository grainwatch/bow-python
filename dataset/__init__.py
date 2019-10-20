from pathlib import Path
from typing import List
import random
import numpy as np

def name_from_path(path: Path) -> str:
    return path.name[:path.name.index('.')]

def shuffle(imgs: List[np.ndarray], labels: List[int]):
    xy = list(zip(imgs, labels))
    random.shuffle(xy)
    return list(zip(*xy))
    
def filter(train_data, test_data, classes):
    train_x, train_y = train
    test_x, test_y = test
    i = 0
    while i < len(train_y):
        if train_y[i] not in classes:
            del train_x[i]
            del train_y[i]
        else:
            i += 1
    i = 0
    while i < len(test_y):
        if test_y[i] not in classes:
            del test_x[i]
            del test_y[i]
        else:
            i += 1
    return (train_x, train_y), (test_x, test_y)