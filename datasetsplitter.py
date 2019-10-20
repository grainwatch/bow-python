from argparse import ArgumentParser
import json
from pathlib import Path
from typing import List
import shutil
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
#import objdataset
from util import rectangle

NEGATIVE_FOLDERNAME = 'negative'

def main():
    parser = ArgumentParser()
    parser.add_argument('imgpath')
    parser.add_argument('labelpath')
    parser.add_argument('dstpath')
    parser.add_argument('scales', metavar='S', type=float, nargs='+')
    parser.add_argument('--selsearch', action='store_true')
    args = parser.parse_args()
    imgpath = Path(args.imgpath)
    labelpath = Path(args.labelpath)
    dstpath = Path(args.dstpath)
    img_classpaths = sorted(imgpath.iterdir())
    label_classpaths = sorted(labelpath.iterdir())
    if NEGATIVE_FOLDERNAME in img_classpaths:
        img_classpaths.remove(NEGATIVE_FOLDERNAME)
    rescale_per_class = args.scales
    do_selsearch = args.selsearch
    split_dataset(img_classpaths, label_classpaths, rescale_per_class, dstpath, do_selsearch)
    

def split_dataset(img_classpaths: List[Path], label_classpaths: List[Path], rescale_per_class: List[float], dstpath: Path, do_selsearch: bool):
    for img_classpath, label_classpath, rescale in zip(img_classpaths, label_classpaths, rescale_per_class):
        bin_edges, count_dict = get_bin_edges(label_classpath)
        label_histogram = create_histogram(bin_edges, count_dict)
        split1_histogram, split2_histogram = split_histogram(label_histogram, 0.2)
        split1_list = histogram_to_list(split1_histogram)
        split2_list = histogram_to_list(split2_histogram)
        create_train_test_split(dstpath, split1_list, split2_list, img_classpath, label_classpath, rescale, do_selsearch)

def get_bin_edges(labelfolder: Path) -> np.ndarray:
    count_dict = {}
    for labelpath in labelfolder.glob('*.json'):
        count = count_labels_in_img(labelpath)
        count_dict[labelpath.name] = count
    count_list = list(count_dict.values())
    histogram, bin_edges = np.histogram(np.asarray(count_list), bins=5)
    return bin_edges, count_dict

def create_histogram(bin_edges: np.ndarray, count_dict):
    histogram = [[] for _ in range(bin_edges.shape[0])]
    last_bin_left = bin_edges[len(histogram)-2]
    last_bin_right = bin_edges[len(histogram)-1]
    for label, count in count_dict.items():
        if last_bin_left <= count <= last_bin_right:
            histogram[len(histogram)-1].append(label)
            continue
        for i in range(len(histogram)-1):
            if bin_edges[i] <= count < bin_edges[i+1]:
                histogram[i].append(label)
    return histogram


def count_labels_in_img(labelpath: Path):
    with labelpath.open() as jsonfile:
        labels_json = json.load(jsonfile)
        labels = labels_json['shapes']
        return len(labels)

def split_histogram(histogram: List[List[str]], split_ratio: float) -> List[List[str]]:
    split1_histogram = []
    split2_histogram = []
    for entry in histogram:
        split_mid = round((1-split_ratio) * len(entry))
        split1_entry = entry[:split_mid]
        split2_entry = entry[split_mid:]
        split1_histogram.append(split1_entry)
        split2_histogram.append(split2_entry)
    return split1_histogram, split2_histogram

def histogram_to_list(histogram):
    histogram_list = []
    for entry in histogram:
        histogram_list += entry
    return histogram_list

def create_train_test_split(dstpath, trainsplit_list, testsplit_list, imgfolder, labelfolder, scale, do_selsearch):
    img_dstpath = dstpath.joinpath('imgs')
    label_dstpath = dstpath.joinpath('labels')
    
    train_imgpath, test_imgpath = img_dstpath.joinpath('train'), img_dstpath.joinpath('test')
    train_labelpath, test_labelpath = label_dstpath.joinpath('train'), label_dstpath.joinpath('test')
    
    train_classimgpath, test_classimgpath = train_imgpath.joinpath(imgfolder.name), test_imgpath.joinpath(imgfolder.name)
    train_classlabelpath, test_classlabelpath = train_labelpath.joinpath(imgfolder.name), test_labelpath.joinpath(imgfolder.name)
    
    train_classimgpath.mkdir(parents=True, exist_ok=True)
    test_classimgpath.mkdir(parents=True, exist_ok=True)
    train_classlabelpath.mkdir(parents=True, exist_ok=True)
    test_classlabelpath.mkdir(parents=True, exist_ok=True)
    
    copy_imgs(trainsplit_list, imgfolder, train_classimgpath, scale)
    copy_imgs(testsplit_list, imgfolder, test_classimgpath, scale)
    copy_labels(trainsplit_list, labelfolder, train_classlabelpath, scale)
    copy_labels(testsplit_list, labelfolder, test_classlabelpath, scale)
    if do_selsearch:
        rect_dstpath = dstpath.joinpath('searchedlabels')
        train_rectpath, test_rectpath = rect_dstpath.joinpath('train'), rect_dstpath.joinpath('test')
        train_classrectpath, test_classrectpath = train_rectpath.joinpath(imgfolder.name), test_rectpath.joinpath(imgfolder.name)
        train_classrectpath.mkdir(parents=True, exist_ok=True)
        test_classrectpath.mkdir(parents=True, exist_ok=True)
        create_rects(trainsplit_list, train_classimgpath, train_classrectpath)
        create_rects(testsplit_list, test_classimgpath, test_classrectpath)

def copy_labels(label_list, srcpath, dstpath, scale):
    for labelname in label_list:
        srclabelpath = srcpath.joinpath(labelname)
        rects, labels = rectangle.rects_labels_from_json(srclabelpath)
        scaled_rects = list(map(lambda rect: rect.rescale(scale), rects))
        dstlabelpath = dstpath.joinpath(labelname)
        rectangle.rects_labels_to_json(scaled_rects, labels, dstlabelpath)

def copy_imgs(label_list, srcpath, dstpath, scale):
    for labelname in label_list:
        i = labelname.index('.')
        imgname = f'{labelname[:i]}.JPG'
        srcimgpath = srcpath.joinpath(imgname)
        img = cv.imread(str(srcimgpath))
        img = cv.resize(img, (0, 0), None, scale, scale)
        dstimgpath = dstpath.joinpath(imgname)
        cv.imwrite(str(dstimgpath), img)

def create_rects(label_list, srcpath, dstpath):
    sel_search = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for labelname in label_list:
        i = labelname.index('.')
        imgname = f'{labelname[:i]}.JPG'
        srcimgpath = srcpath.joinpath(imgname)
        img = cv.imread(str(srcimgpath))
        rects = search_rects(sel_search, img)
        dstrectpath = dstpath.joinpath(labelname)
        rectangle.rects_to_json(rects, dstrectpath)

def search_rects(sel_search, img):
    sel_search.setBaseImage(img)
    sel_search.switchToSelectiveSearchFast()
    sel_rects = sel_search.process()
    sel_rects = list(map(lambda rect: rectangle.Rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])), sel_rects))
    return sel_rects


if __name__ == '__main__':
    main()
