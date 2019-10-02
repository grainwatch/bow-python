from typing import Tuple, List, Dict
from numpy import ndarray
from pathlib import Path
import json
import cv2 as cv
import numpy as np
import random
import math

Data = Tuple[List[ndarray], List[str]]
Position = Tuple[int, int]
Size = Position

label_to_id = {'negative': -1, 'corn': 1, 'rye': 2, 'triticale': 3, 'wheat': 4}

def load_data(data_pathname) -> Data:
    data_path = Path(data_pathname)
    labels_path = data_path.joinpath('labels.csv')
    csvfile = labels_path.open()
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    y = next(reader)

    x = []
    for img_path in data_path.glob('*.jpg'):
        img = cv.imread(str(img_path))
        x.append(img)
    return x, y

def load_data_json(data_pathname) -> Tuple[Data, Data]:
    data_path = Path(data_pathname)
    labels_path = data_path.joinpath('labels.json')
    jsonfile = labels_path.open()
    label_dict = json.load(jsonfile)
    train_x, train_y, test_x, test_y = [], [], [], []
    for label in label_dict.keys():
        train_boundaries = label_dict[label]['train_boundaries']
        test_boundaries = label_dict[label]['test_boundaries']
        label_id = label_to_id[label]

        label_train_x = _read_imgs(data_path, train_boundaries)
        train_x += label_train_x
        train_y += [label_to_id[label]] * len(label_train_x)

        label_test_x = _read_imgs(data_path, test_boundaries)
        test_x += label_test_x
        test_y += [label_to_id[label]] * len(label_test_x)
    return (train_x, train_y), (test_x, test_y)

def load_data_2(data_pathname) -> Tuple[Data, Data]:
    imgs = []
    labels = []
    data_path = Path(data_pathname)
    train_path = data_path.joinpath('train')
    test_path = data_path.joinpath('test')
    assert train_path.exists() and test_path.exists()
    train_imgs, train_labels = _read_imgs_from_classfolder(train_path)
    test_imgs, test_labels = _read_imgs_from_classfolder(train_path)
    return (train_imgs, train_labels), (test_imgs, test_labels)


def _read_imgs_from_classfolder(classdata_path: Path):
    imgs = []
    labels = []
    for trainclass_path in classdata_path.iterdir():
        trainimg_pathlist = list(trainclass_path.iterdir())
        classimgs = list(map(lambda imgpath: cv.imread(str(imgpath)), trainimg_pathlist))
        classlabels = [label_to_id[trainclass_path.name]] * len(classimgs)
        imgs += classimgs
        labels += classlabels
    return imgs, labels


def _read_imgs(data_path, boundary):
    x = []
    for i in range(boundary[0], boundary[1] + 1):
            img_path = data_path.joinpath(f'{i}.jpg')
            img = cv.imread(str(img_path))
            x.append(img)
    return x
            

def shuffle(x: List[np.ndarray], y: List[int]):
    xy = list(zip(x, y))
    random.shuffle(xy)
    return list(zip(*xy))
    
def filter_data(train, test, classes):
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

def _in_class_list(classes, xy):
    print(f'')

GER_TO_ENG = {'mais': 'corn', 'roggen': 'rye', 'trictale': 'trictale', 'weizen': 'wheat'}
ENG_TO_GER = {'corn': 'mais', 'rye': 'roggen', 'triticale': 'triticale', 'wheat': 'weizen'}
class_dict = {'mehl': -1, 'mais': 1, 'roggen': 2, 'triticale': 3, 'weizen': 4}
class_dict = {'mehl': -1, 'mais': 1, 'roggen': 2, 'triticale': 3, 'weizen': 4}

def _create_negative_examples(img_pathname: str, negative_foldername: str, dst_pathname: str, amount: int, scale: float, window_size: Tuple[int, int], img_counter):
    under_boundary = img_counter
    dst_path = Path(dst_pathname)
    negative_folderpath = Path(img_pathname).joinpath(negative_foldername)
    examples_created = 0
    current_x = 0
    current_y = 0
    imgs = negative_folderpath.glob('*.jpg')
    img = cv.imread(str(next(imgs)))
    img = cv.resize(img, (0, 0), None, scale, scale)
    while examples_created < amount:
        if current_x + window_size[0] > img.shape[1]:
            current_x = 0
            current_y += window_size[1]
        if current_y + window_size[1] > img.shape[0]:
            current_x = 0
            current_y = 0
            img = cv.imread(str(next(imgs)))
            img = cv.resize(img, (0, 0), None, scale, scale)
        roi_img = img[current_y:current_y+window_size[1], current_x:current_x+window_size[0]]
        roi_path = dst_path.joinpath(f'{img_counter}.jpg')
        cv.imwrite(str(roi_path), roi_img)
        img_counter += 1
        current_x += window_size[0]
        examples_created += 1
    upper_boundary = img_counter - 1
    return under_boundary, upper_boundary

### VERSION 2 ###

def create_data_2(img_pathname: str, label_pathname: str, dst_pathname: str='.', test_amount: float=0, scale: float=1.0, window_size: Tuple[int, int]=(100, 100)):
    dst_path = Path(dst_pathname)
    dst_path.mkdir(exist_ok=True)
    train_path = dst_path.joinpath('train')
    test_path = dst_path.joinpath('test')
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    negative_samples = []
    for label_classdir in Path(label_pathname).iterdir():
        img_classdir = Path(img_pathname).joinpath(label_classdir.name)
        if img_classdir.exists():
            samples = _create_class_samples(img_classdir, label_classdir, scale, window_size)
            negative_samples += _create_negative_samples_from_class(img_classdir, label_classdir, scale, window_size)
            train_samples, test_samples = _split_samples(samples, test_amount)
            class_train_path = train_path.joinpath(label_classdir.name)
            class_train_path.mkdir(exist_ok=True)
            class_test_path = test_path.joinpath(label_classdir.name)
            class_test_path.mkdir(exist_ok=True)
            _write_samples(train_samples, class_train_path)
            _write_samples(test_samples, class_test_path)
    negativetrain_path = train_path.joinpath('negative')
    negativetrain_path.mkdir(exist_ok=True)
    negativetest_path = test_path.joinpath('negative')
    negativetest_path.mkdir(exist_ok=True)
    negativetrain_samples, negativetest_samples = _split_samples(negative_samples, test_amount)
    _write_samples(negativetrain_samples, negativetrain_path)
    _write_samples(negativetest_samples, negativetest_path)

def create_data_from_classes(imgfolder_pathname: str, labelfolder_pathname: str, dst_pathname: str, class_list: List[str], test_amount: float=0, scale: float=1.0, window_size: Tuple[int, int]=(128, 128)):
    dst_path = Path(dst_pathname)
    dst_path.mkdir(exist_ok=True)
    train_path = dst_path.joinpath('train')
    test_path = dst_path.joinpath('test')
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    labelpaths = list(Path(labelfolder_pathname).iterdir())
    labelpaths = list(filter(lambda path: path.name in class_list, labelpaths))
    negative_samples = []
    for label_classdir in labelpaths:
        img_classdir = Path(imgfolder_pathname).joinpath(label_classdir.name)
        if img_classdir.exists():
            samples = _create_class_samples(img_classdir, label_classdir, scale, window_size)
            negative_samples += _create_negative_samples_from_class(img_classdir, label_classdir, scale, window_size)
            train_samples, test_samples = _split_samples(samples, test_amount)
            class_train_path = train_path.joinpath(label_classdir.name)
            class_train_path.mkdir(exist_ok=True)
            class_test_path = test_path.joinpath(label_classdir.name)
            class_test_path.mkdir(exist_ok=True)
            _write_samples(train_samples, class_train_path)
            _write_samples(test_samples, class_test_path)
    negativetrain_path = train_path.joinpath('negative')
    negativetrain_path.mkdir(exist_ok=True)
    negativetest_path = test_path.joinpath('negative')
    negativetest_path.mkdir(exist_ok=True)
    negativetrain_samples, negativetest_samples = _split_samples(negative_samples, test_amount)
    _write_samples(negativetrain_samples, negativetrain_path)
    _write_samples(negativetest_samples, negativetest_path)
    

def _write_samples(samples: List[np.ndarray], dstpath: Path):
    i = 0
    for sample in samples:
        samplepath = dstpath.joinpath(f'{i}.jpg')
        cv.imwrite(str(samplepath), sample)
        i += 1

#TESTED
def _split_samples(samples: List[np.ndarray], test_amount: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    train_amount =  math.floor((1 - test_amount) * len(samples))
    train_samples = samples[:train_amount]
    test_samples = samples[train_amount:]
    return train_samples, test_samples

def _create_class_samples(classimg_path: Path, labelfolder_path: Path, scale: float, window_size: Tuple[int, int]) -> List[np.ndarray]:
    samples = []
    window_dict = _load_labels_from_path(list(labelfolder_path.glob('*.json')), window_size, True)
    imgpaths = list(classimg_path.iterdir())
    imgpaths = list(filter(lambda imgpath: _name_from_path(imgpath) in window_dict, imgpaths))
    labels = list(map(lambda imgpath: window_dict[_name_from_path(imgpath)], imgpaths))
    imgs = _load_imgs_from_path(imgpaths, scale)
    for img, label in zip(imgs, labels):
        label = list(map(lambda roi: _move_window_into_img(roi, img.shape), label))
        samples += _samples_from_img(img, label)
    return samples

def _create_negative_samples_from_class(classimg_path: Path, labelfolder_path: Path, scale: float, window_size: Tuple[int, int]) -> List[np.ndarray]:
    samples = []
    labelpaths = list(labelfolder_path.glob('*.json'))
    roi_dict = _load_labels_from_path(labelpaths)
    imgpaths = list(classimg_path.glob('*.jpg'))
    imgpaths = list(filter(lambda imgpath: _name_from_path(imgpath) in roi_dict, imgpaths))
    labels = map(lambda imgpath: roi_dict[_name_from_path(imgpath)], imgpaths)
    imgs = _load_imgs_from_path(imgpaths, scale)
    for img, label in zip(imgs, labels):
        samples += _negative_samples_from_img(img, label)
    return samples

    
Rectangle = Tuple[int, int, int, int]

def _load_labels_from_path(labelpaths: List[Path], window_size: Tuple[int, int]=None, ignore_none: bool=False) -> Dict[str, List[Rectangle]]:
    window_dict = {}
    for labelpath in labelpaths:
        labelname = _name_from_path(labelpath)
        rois = _rois_from_json(labelpath, ignore_none)
        rois = map(lambda roi: _scale_rectangle(roi, scale), rois)
        if window_size is not None:
            rois = map(lambda roi: _rectangle_to_window(roi, window_size), rois)
        window_dict[labelname] = list(rois)
    return window_dict

def _move_window_into_img(roi: Rectangle, img_shape: Tuple[int, int]):
    x, y, width, height = roi
    img_width, img_height = img_shape[1], img_shape[0]
    if x < 0:
        x = 0
    elif x + width > img_width:
        x = img_width - width
    if y < 0:
        y = 0
    elif y + height > img_height:
        y = img_height - height
    return x, y, width, height

#TESTED
def _name_from_path(path: Path) -> str:
    return path.name[:path.name.index('.')]

def _load_imgs_from_path(imgpaths: List[Path], scale: float=1.0):
    imgs = map(lambda imgpath: cv.imread(str(imgpath)), imgpaths)
    if scale != 1.0:
        imgs = map(lambda img: cv.resize(img, (0, 0), None, scale, scale), imgs)
    return list(imgs)

#TESTED
def _samples_from_img(img: np.ndarray, rois: List[Rectangle]) -> List[np.ndarray]:
    samples = []
    for roi in rois:
        x, y, width, height = roi
        roi_img = img[y:y+height, x:x+width]
        samples.append(roi_img)
    return samples

def _negative_samples_from_img(img: np.ndarray, rois: List[Rectangle]=None, window_size: Tuple[int, int]=(128, 128), stride: Tuple[int, int]=(192, 192)):
    roi_img_list = []
    width, height = window_size
    stride_x, stride_y = stride
    x, y = 0, 0
    while y + height < img.shape[0]:
        if x + width > img.shape[1]:
            x = 0
            y += stride_y
            continue #skip loop iteration to check y condition
        if not is_roi_colliding((x, y, width, height), rois):
            roi_img = img[y:y+height, x:x+width]
            roi_img_list.append(roi_img)
        x += stride_x
    return roi_img_list

#TESTED
def is_roi_colliding(roi_to_check: Rectangle, rois: List[Rectangle], img: np.ndarray=None):
    x_check, y_check, width_check, height_check = roi_to_check
    #visual = cv.rectangle(img, (x_check, y_check), (x_check+width_check, y_check+height_check), (255, 0, 0))
    for roi in rois:
        x, y, width, height = roi
        x_condition = (x_check >= x and x_check < x + width) or (x_check + width_check > x and x_check + width_check <= x + width) or (x_check <= x and x_check + width_check >= x + width)
        y_condition = (y_check >= y and y_check < y + height) or (y_check + height_check > y and y_check + height_check <= y + height) or (y_check <= y and y_check + height_check >= y + height)
        if x_condition and y_condition:
            #if img is not None:
                #visual = cv.rectangle(visual, (x, y), (x+width, y+height), (0, 255, 0))
                #cv.imshow('', visual)
                #cv.waitKey(0)
            return True
        #visual = cv.drawText()
        #visual = cv.rectangle(visual, (x, y), (x+width, y+height), (0, 0, 255))
    #cv.imshow('', visual)
    #cv.waitKey(0)
    return False

#TESTED
def _rois_from_json(jsonpath: Path, ignore_none: bool=False):
    rois = []
    with jsonpath.open() as jsonfile:
        labeljson_dict = json.load(jsonfile)
        labels = labeljson_dict['shapes']
        for roi_label in labels:
            points = roi_label['points']
            rectangle = _rectangle_from_points(points)
            if ignore_none:
                if roi_label['label'] != 'None':
                    rois.append(rectangle)
            else:
                rois.append(rectangle)
    return rois

#TESTED
def _rectangle_from_points(points):
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
    return x[0], y[0], width, height

#TESTED
def _scale_rectangle(roi: Rectangle, scale: float) -> Rectangle:
    return tuple(map(lambda p: int(p*scale), roi))

#TESTED
def _rectangle_to_window(roi: Rectangle, window_size: Tuple[int, int]) -> Rectangle:
    x, y, width, height = roi
    win_width, win_height = window_size
    center_x = x + width/2
    center_y = y + height/2
    win_x = int(center_x - win_width/2)
    win_y = int(center_y - win_height/2)
    return (win_x, win_y, win_width, win_height)


### OLD ###

def calc_roi(points, window_size, scale, img_shape):
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
    label_width = x[1] - x[0]
    label_height = y[1] - y[0]
    center_x = round((x[0] + label_width/2) * scale)
    center_y = round((y[0] + label_height/2) * scale)
    x[0] = center_x - int(window_size[0]/2)
    x[1] = center_x + int(window_size[0]/2)
    y[0] = center_y - int(window_size[1]/2)
    y[1] = center_y + int(window_size[1]/2)
    if x[0] < 0:
        x[1] += abs(x[0])
        x[0] = 0
    elif x[1] > img_shape[1]:
        x[0] -= (x[1] - img_shape[1])
        x[1] = img_shape[1]
    if y[0] < 0:
        y[1] += abs(y[0])
        y[0] = 0
    elif y[1] > img_shape[0]:
        y[0] -= (y[1] - img_shape[0])
        y[1] = img_shape[0]
    return x, y

        
def get_biggest_label(label_pathname: str):
    size = 0
    for classdir in Path(label_pathname).iterdir():
        for labelfile in classdir.glob('*.json'):
            print(labelfile)
            labels = json.load(labelfile.open())['shapes']
            for label in labels:
                points = label['points']
                label_width = abs(points[0][0] - points[1][0])
                label_height = abs(points[0][1] - points[1][1])
                if label_width > size:
                    size = label_width
                if label_height > size:
                    size = label_height
    return size                   

def split_img_in_windows(img: np.ndarray, window_size: Size, window_stride: Size) -> Tuple[List[np.ndarray], List[Position]]:
    imgs = []
    img_pos_list = []
    window_pos = [0, 0]
    while window_pos[1] + window_size[1] < img.shape[0]:
        roi = img[window_pos[1]:window_pos[1]+window_size[1], window_pos[0]:window_pos[0]+window_size[0]]
        imgs.append(roi)
        img_pos_list.append(tuple(window_pos.copy()))
        window_pos[0] += window_stride[0]
        if window_pos[0] + window_size[0] > img.shape[1]:
            window_pos[0] = 0
            window_pos[1] += window_stride[1]
    return imgs, img_pos_list

def save_img_splits(dst_folder: str, imgs: List[np.ndarray], img_pos_list: List[Position]):
    assert len(imgs) == len(img_pos_list)
    pos_file_dict = {}
    dst_folderpath = Path(dst_folder)
    
    dst_folderpath.mkdir(exist_ok=True)
    for i in range(len(imgs)):
        img = imgs[i]
        split_filename = str(dst_folderpath.joinpath(f'{i}.jpg'))
        pos_file_dict[i] = img_pos_list[i]
        result = cv.imwrite(split_filename, img)
    with dst_folderpath.joinpath('windows.json').open(mode='w') as jsonfile:
        json.dump(pos_file_dict, jsonfile)



if __name__ == '__main__':
    size = get_biggest_label('C:/Users/Alex/IdeaProjects/grain-swpt/dataset')
    scale = 128 / size
    print(scale)
    create_data_from_classes('C:/Users/Alex/Desktop/Bilder_Korner_original_20180411', 'C:/Users/Alex/IdeaProjects/grain-swpt/dataset2', 'dataset_v2', ['corn'], 0.2, scale, (128, 128))
    """(train_x, train_y), (test_x, test_y) = load_data_json('dataset')
    print(train_x, train_y, test_x, test_y)"""
    #x, y = load_data('./dataset')
    #x, y = load_2_class('dataset', 0, 1)
    #print(x, y)