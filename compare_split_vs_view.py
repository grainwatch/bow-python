from argparse import ArgumentParser
import cv2 as cv
import dataset
import model
import json
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('imgpath')
parser.add_argument('splitpath')
parser.add_argument('-w', '--window', type=int, default=128)
parser.add_argument('-z', '--scale', type=float, default=1.0)
args = parser.parse_args()

splitpath = Path(args.splitpath)

origin_img = cv.imread(args.imgpath)
origin_img = cv.resize(origin_img, (0, 0), None, args.scale, args.scale)

hogmodel = model.HOG.load('hog', 'svm')

with splitpath.joinpath('windows.json').open(mode='r') as jsonfile:
    counter = 0

    winpos_dict = json.load(jsonfile)

    img_paths = list(splitpath.glob('*.jpg'))
    assert len(img_paths) == len(winpos_dict)
    for split_filepath in img_paths:
        splitimg = cv.imread(str(split_filepath))

        i = split_filepath.name[:split_filepath.name.index('.')]

        winpos = winpos_dict[i]
        win = (slice(winpos[1], winpos[1] + args.window), slice(winpos[0], winpos[0] + args.window))
        sliceimg = origin_img[win]

        response = hogmodel.predict([splitimg, sliceimg])
        if response[0] != response[1]:
            counter += 1
    print(f'counter: {counter}')
    print(f'ratio: {counter/ len(img_paths)}')        