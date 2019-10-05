from argparse import ArgumentParser
from pathlib import Path
import json
import cv2 as cv
import dataset
import model

parser = ArgumentParser()
parser.add_argument('imgpath')
parser.add_argument('splitpath')
parser.add_argument('-z', '--scale', type=float, default=1.0)
args = parser.parse_args()

splitpath = Path(args.splitpath)

origin_img = cv.imread(args.imgpath)
origin_img = cv.resize(origin_img, (0, 0), None, args.scale, args.scale)

hogmodel = model.HOG.load('hog', 'svm')

with splitpath.joinpath('windows.json').open(mode='r') as jsonfile:
    valid_winposes = []
    winpos_dict = json.load(jsonfile)

    img_paths = sorted(splitpath.glob('*.jpg'))
    for x in img_paths:
        filename = str(x)
        img = cv.imread(filename)
        response = hogmodel.predict([img])
        res = response[0]
        if res == 1:
            imgname = x.name
            i = imgname[:imgname.index('.')]
            valid_winposes.append(winpos_dict[i])

    for pos in valid_winposes:
        origin_img = cv.rectangle(origin_img, (pos[0], pos[1], 128, 128), (0, 255, 0))
    cv.imshow('', origin_img)
    cv.waitKey(0)