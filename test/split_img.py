from argparse import ArgumentParser
import cv2 as cv
import dataset

parser = ArgumentParser()
parser.add_argument('imgname')
parser.add_argument('dstfolder')
parser.add_argument('-w', '--window', type=int, default=128)
parser.add_argument('-s', '--stride', type=int, default=64)
parser.add_argument('-z', '--scale', type=float, default=1.0)
args = parser.parse_args()

img = cv.imread(args.imgname)
img = cv.resize(img, (0, 0), None, args.scale, args.scale)

window_size = (args.window, args.window)
window_stride = (args.stride, args.stride)

img_splits, split_pos_list = dataset.split_img_in_windows(img, window_size, window_stride)
dataset.save_img_splits(args.dstfolder, img_splits, split_pos_list)

