from typing import Tuple, List
from numpy import ndarray
from pathlib import Path
import json
import cv2 as cv
import numpy as np
import random
import math

Data = Tuple[List[ndarray], List[str]]

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


def _read_imgs(data_path, boundary):
    x = []
    for i in range(boundary[0], boundary[1] + 1):
            img_path = data_path.joinpath(f'{i}.jpg')
            img = cv.imread(str(img_path))
            x.append(img)
    return x
            


def load_2_class(data_pathname:str, class1: int, class2: int) -> Tuple[List[ndarray], List[str]]:
    data_path = Path(data_pathname)
    labels_path = data_path.joinpath('labels.csv')
    csvfile = labels_path.open()
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    y = next(reader)
    img_ids = [(i, class_id) for i, class_id in enumerate(y) if class_id == class1 or class_id == class2]
    y = []
    x = []
    for i, class_id in img_ids:
        y.append(class_id)
        img_path = data_path.joinpath(f'{i}.jpg')
        x.append(cv.imread(str(img_path)))
    return x, y

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
    print(train_y)
    print(test_y)
    """train_xy = list(zip(train[0], train[1]))
    test_xy = list(zip(test[0], test[1]))
    filter(lambda x: x[1] in classes, train_xy)
    i = 0
    while i < len(train_xy):
        if train_xy[i][1] not in classes:
            del train_xy[i]
        i += 1
    print(test_xy)
    train_x, train_y = zip(*train_xy)
    test_x, test_y = zip(*test_xy)
    print(train_y)
    print(test_y)"""
    return (train_x, train_y), (test_x, test_y)

def _in_class_list(classes, xy):
    print(f'')

GER_TO_ENG = {'mais': 'corn', 'roggen': 'rye', 'trictale': 'trictale', 'weizen': 'wheat'}
ENG_TO_GER = {'corn': 'mais', 'rye': 'roggen', 'triticale': 'triticale', 'wheat': 'weizen'}
class_dict = {'mehl': -1, 'mais': 1, 'roggen': 2, 'triticale': 3, 'weizen': 4}
class_dict = {'mehl': -1, 'mais': 1, 'roggen': 2, 'triticale': 3, 'weizen': 4}

def create_data(img_pathname: str, label_pathname: str, negative_foldername: str, dst_pathname: str='.', test_amount: float=0, scale: float=None, window_size: Tuple[int, int]=(100, 100)):
        dst_path = Path(dst_pathname)
        dst_path.mkdir(exist_ok=True)

        missing_files_dict = {}
        label_dict = {}
        img_counter = 0
        
        for label_classdir in Path(label_pathname).iterdir():
            img_classdir = Path(img_pathname).joinpath(ENG_TO_GER[label_classdir.name])
            
            if img_classdir.exists():
                missing_label_counter = 0
                missing_label_names = []
                under_boundary = img_counter
                for img_path in img_classdir.iterdir():
                    i = img_path.name.index('.')
                    file_id = img_path.name[0:i]
                    label_path = label_classdir.joinpath(file_id + '.json')

                    if label_path.exists():
                        img = cv.imread(str(img_path))
                        if scale is not None:
                            img = cv.resize(img, (0, 0), None, scale, scale)
                        labels = json.load(label_path.open())['shapes']
                        
                        for label in labels:
                            points = label['points']
                            x, y = calc_roi(points, window_size, scale, img.shape)
                            item = img[y[0]:y[1], x[0]:x[1]]
                            item_path = dst_path.joinpath(f'{img_counter}.jpg')
                            cv.imwrite(str(item_path), item)
                            img_counter +=1
                    
                    else:
                        missing_label_counter += 1
                        missing_label_names.append(img_path.name)
                
                upper_boundary = img_counter - 1
                train_boundaries, test_boundaries = _calc_train_test_boundaries(under_boundary, upper_boundary, test_amount)
                label_dict[label_classdir.name] = {'train_boundaries': train_boundaries, 'test_boundaries': test_boundaries}

                missing_files_dict[label_classdir.name] = (missing_label_counter, missing_label_names)
        print(missing_files_dict)
        under_boundary, upper_boundary = _create_negative_examples(img_pathname, negative_foldername, dst_pathname, 400, scale, window_size, img_counter)
        train_boundaries, test_boundaries = _calc_train_test_boundaries(under_boundary, upper_boundary, test_amount)
        label_dict['negative'] = {'train_boundaries': train_boundaries, 'test_boundaries': test_boundaries}
        with dst_path.joinpath('labels.json').open('w') as jsonfile:
            json.dump(label_dict, jsonfile)

def _calc_train_test_boundaries(under_boundary, upper_boundary, test_imgs_amount):
    print(under_boundary, upper_boundary, test_imgs_amount)
    total = upper_boundary - under_boundary
    test_imgs_total = math.ceil(total*test_imgs_amount)
    under_train_boundary = under_boundary
    upper_train_boundary = upper_boundary - test_imgs_total
    under_test_boundary = upper_train_boundary + 1
    upper_test_boundary = upper_boundary
    return (under_train_boundary, upper_train_boundary), (under_test_boundary, upper_test_boundary)

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


if __name__ == '__main__':
    size = get_biggest_label('C:/Users/Alex/IdeaProjects/grain-swpt/dataset')
    scale = 128 / size
    print(scale)
    #create_data('C:/Users/Alex/Desktop/Bilder_Korner_original_20180411', 'C:/Users/Alex/IdeaProjects/grain-swpt/dataset', 'mehl', 'dataset', 0.2, scale, (128, 128))
    (train_x, train_y), (test_x, test_y) = load_data_json('dataset')
    print(train_x, train_y, test_x, test_y)
    #x, y = load_data('./dataset')
    #x, y = load_2_class('dataset', 0, 1)
    #print(x, y)