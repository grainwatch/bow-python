from pathlib import Path
from typing import Tuple, List
import json

import numpy as np
import cv2 as cv

import util.rectangle as rectangle
import dataset

LABELNAME_TO_LABELID = {'negative': -1, 'corn': 1, 'rye': 2, 'triticale': 3, 'wheat': 4, 'half_corn': 5, 'half_rye': 6, 'half_triticale': 7, 'half_wheat': 8}

Dataset = Tuple[Tuple[List[np.ndarray], List[rectangle.Rect], List[int]], Tuple[List[np.ndarray], List[rectangle.Rect], List[int]]]

def load_grain_dataset(dataset_pathname: str, folders=['corn', 'rye', 'triticale', 'wheat'], with_searched_labels: bool = False, ignore_train_half_labels: bool = True) -> Dataset:
    datasetpath = Path(dataset_pathname)
    imgfolderspath = datasetpath.joinpath('imgs')
    labelfolderpath = datasetpath.joinpath('labels')

    train_labelfolderpath = labelfolderpath.joinpath('train')
    test_labelfolderpath = labelfolderpath.joinpath('test')
    assert train_labelfolderpath.exists() and test_labelfolderpath.exists()

    train_imgfolderpath = imgfolderspath.joinpath('train')
    test_imgfolderpath = imgfolderspath.joinpath('test')
    assert train_imgfolderpath.exists() and test_imgfolderpath.exists()

    train_img_dict = _load_img_dict(train_imgfolderpath)
    test_img_dict = _load_img_dict(test_imgfolderpath)
    train_rect_dict = _load_rects(train_labelfolderpath, ignore_half_labels=ignore_train_half_labels)
    test_rect_dict = _load_rects(test_labelfolderpath)
    

    if with_searched_labels:
        searchedfolderpath = datasetpath.joinpath('searchedlabels')
        train_searchedfolderpath = searchedfolderpath.joinpath('train')
        test_searchedfolderpath = searchedfolderpath.joinpath('test')
        assert train_searchedfolderpath.exists() and test_searchedfolderpath.exists()
        train_searched_dict = _load_rects(train_searchedfolderpath, labeled=False)
        test_searched_dict = _load_rects(test_searchedfolderpath, labeled=False)
        train_imgs, train_rects, train_labels, train_searched = dataset_from_img_dicts(train_img_dict, train_rect_dict, folders, train_searched_dict)
        test_imgs, test_rects, test_labels, test_searched = dataset_from_img_dicts(test_img_dict, test_rect_dict, folders, test_searched_dict)
        return (train_imgs, train_rects, train_labels, train_searched), (test_imgs, test_rects, test_labels, test_searched)
    else:
        train_imgs, train_rects, train_labels = dataset_from_img_dicts(train_img_dict, train_rect_dict, folders)
        test_imgs, test_rects, test_labels = dataset_from_img_dicts(test_img_dict, test_rect_dict, folders)
        return (train_imgs, train_rects, train_labels), (test_imgs, test_rects, test_labels)


def dataset_from_img_dicts(imgclass_dict, rectclass_dict, folders=['corn', 'rye', 'triticale', 'wheat'], searchedclass_dict=None):
    imgs, img_rects, img_labels, img_searched_rects = [], [], [], []
    for foldername in folders:
        rect_dict = rectclass_dict[foldername]
        img_dict = imgclass_dict[foldername]
        if searchedclass_dict is not None:
            searched_dict = searchedclass_dict[foldername]
        for imgname in rect_dict.keys():
            rects, labels = rect_dict[imgname]
            labels = list(map(lambda label: LABELNAME_TO_LABELID[label], labels))
            imgs.append(img_dict[imgname])
            img_rects.append(rects)
            img_labels.append(labels)
            if searchedclass_dict is not None:
                img_searched_rects.append(searched_dict[imgname])
    if searchedclass_dict is None:
        return imgs, img_rects, img_labels
    else:
        return imgs, img_rects, img_labels, img_searched_rects


def _load_rects(rectfolderpath: Path, labeled=True, ignore_half_labels=False):
    rect_dict = {}
    for grainfolderpath in rectfolderpath.iterdir():
        rectclass_dict = {}
        for rectpath in grainfolderpath.glob('*.json'):
            imgname = dataset.name_from_path(rectpath)
            if labeled:
                rects, labels = rectangle.rects_labels_from_json(rectpath)
                if ignore_half_labels:
                    rects, labels = _filter_half_labels(rects, labels)
                rectclass_dict[imgname] = (rects, labels)
            else:
                rectclass_dict[imgname] = rectangle.rects_from_json(rectpath)
        rect_dict[grainfolderpath.name] = rectclass_dict
    return rect_dict

def _load_img_dict(imgfolderpath: Path):
    img_dict = {}
    for grainfolderpath in imgfolderpath.iterdir():
        imgclass_dict = {}
        for imgpath in grainfolderpath.glob('*.jpg'):
            imgname = dataset.name_from_path(imgpath)
            imgclass_dict[imgname] = cv.imread(str(imgpath))
        img_dict[grainfolderpath.name] = imgclass_dict
    return img_dict

def _filter_half_labels(rects: List[rectangle.Rect], labels: List[str]) -> Tuple[List[rectangle.Rect], List[str]]:
    filtered_rects_labels = filter(lambda rect_label: rect_label[1] != 'half_corn', zip(rects, labels))
    rects, labels = list(zip(*filtered_rects_labels))
    return rects, labels