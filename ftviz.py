import argparse
from pathlib import Path
import random

import cv2 as cv
import numpy as np

import models
import viz

def viz_fts(debugpath: Path):
    pexamp = debugpath.joinpath('pexamp')
    nexamp = debugpath.joinpath('nexamp')
    p_rois = list(map(lambda imgpath: cv.imread(str(imgpath)), pexamp.glob('*.jpg')))
    n_rois = list(map(lambda imgpath: cv.imread(str(imgpath)), nexamp.glob('*.jpg')))
    p_labels = [1] * len(p_rois)
    n_labels = [-1] * len(n_rois)
    bowmodel = models.BOWModel()
    bowmodel.fit(p_rois + n_rois, p_labels + n_labels)
    p_samples = bowmodel.x_histogram[:len(p_rois)]
    n_samples = bowmodel.x_histogram[len(p_rois):]
    r_samples = random.sample(list(n_samples), 500)
    labels = p_labels + [-1] * len(r_samples)
    fts = np.asarray(list(p_samples) + r_samples)
    viz.visualize(fts, labels)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('debugpath')
    args = parser.parse_args()
    debugpath = Path(args.debugpath)
    viz_fts(debugpath)