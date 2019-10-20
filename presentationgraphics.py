from pathlib import Path
import cv2 as cv
import models
import dataclasses
import util.rectangle as rectangle
from skimage.feature import hog
from skimage import exposure

dstpath = Path('C:\\Users\\Alex\\Desktop\\berichtimgs')
sift = cv.xfeatures2d.SIFT_create()
matcher = cv.BFMatcher_create()

#Feature image
cornimgpath = Path('C:\\Users\\Alex\\CodeProjects\\bow-python\\imgdataset\\3.jpg')
cornimg = cv.imread(str(cornimgpath))
kp, ds = sift.detectAndCompute(cornimg, None)
ftimg = cv.drawKeypoints(cornimg, kp, None)
cv.imwrite(str(dstpath.joinpath('ftimg.png')), ftimg)

#corn
cornimgpath256 = Path('C:\\Users\\Alex\\Desktop\\corn256\\train\\corn\\3.jpg')
cornimg256 = cv.imread(str(cornimgpath256))
b_kp, b_ds = sift.detectAndCompute(cornimg256, None)
b_ftimg = cv.drawKeypoints(cornimg256, b_kp, None, (0, 0, 255))
_, hog_cornimg = hog(cornimg256, pixels_per_cell=(64, 64), cells_per_block=(2, 2), visualize=True)
hog_cornimg = exposure.rescale_intensity(hog_cornimg, in_range=(0, 8), out_range=(0, 255))
cv.imwrite(str(dstpath.joinpath('corn_ftimg.png')), b_ftimg)
cv.imwrite(str(dstpath.joinpath('corn_hogimg.png')), hog_cornimg)


#rye
ryeimgpath256 = Path('C:\\Users\\Alex\\Desktop\\all256\\train\\rye\\5.jpg')
ryeimg256 = cv.imread(str(ryeimgpath256))
rye_kp, rye_ds = sift.detectAndCompute(ryeimg256, None)
rye_ftimg = cv.drawKeypoints(ryeimg256, rye_kp, None, (0, 0, 255))
_, hog_ryeimg = hog(ryeimg256, pixels_per_cell=(64, 64), cells_per_block=(2, 2), visualize=True)
hog_ryeimg = exposure.rescale_intensity(hog_ryeimg, in_range=(0, 8), out_range=(0, 255))
cv.imwrite(str(dstpath.joinpath('rye_ftimg.png')), rye_ftimg)
cv.imwrite(str(dstpath.joinpath('rye_hogimg.png')), hog_ryeimg)

#triticale
triticaleimgpath = Path('C:\\Users\\Alex\\Desktop\\all256\\train\\triticale\\300.jpg')
triticaleimg256 = cv.imread(str(triticaleimgpath))
triti_kp, triti_ds = sift.detectAndCompute(triticaleimg256, None)
triti_ftimg = cv.drawKeypoints(triticaleimg256, triti_kp, None, (0, 0, 255))
_, hog_tritiimg = hog(triticaleimg256, pixels_per_cell=(64, 64), cells_per_block=(2, 2), visualize=True)
hog_tritiimg = exposure.rescale_intensity(hog_tritiimg, in_range=(0, 8), out_range=(0, 255))
cv.imwrite(str(dstpath.joinpath('triti_ftimg.png')), triti_ftimg)
cv.imwrite(str(dstpath.joinpath('triti_hogimg.png')), hog_tritiimg)

#wheat
wheatimgpath = Path('C:\\Users\\Alex\\Desktop\\all256\\train\\wheat\\314.jpg')
wheatimg256 = cv.imread(str(wheatimgpath))
wheat_kp, wheat_ds = sift.detectAndCompute(wheatimg256, None)
wheat_ftimg = cv.drawKeypoints(wheatimg256, wheat_kp, None, (0, 0, 255))
_, hog_wheatimg = hog(wheatimg256, pixels_per_cell=(64, 64), cells_per_block=(2, 2), visualize=True)
hog_wheatimg = exposure.rescale_intensity(hog_wheatimg, in_range=(0, 8), out_range=(0, 255))
cv.imwrite(str(dstpath.joinpath('wheat_ftimg.png')), wheat_ftimg)
cv.imwrite(str(dstpath.joinpath('corn_wheatimg.png')), hog_wheatimg)

matches = matcher.match(b_ds, ds)
matchimg = cv.drawMatches(cornimg256, b_kp, cornimg, kp, matches, None)
cv.imwrite(str(dstpath.joinpath('matchimg.png')), matchimg)

#Color DBScan
fullimg = cv.imread('C:\\Users\\Alex\\Desktop\\Bilder_Korner_original_20180411\\corn\\DSC_0922.JPG')
lower = (0, 120, 150)
upper = (80, 255, 255)
img = cv.resize(fullimg, (0, 0), None, 0.1, 0.1)
resized = img.copy()
tresholdimg = cv.inRange(img.copy(), lower, upper)
cv.imwrite(str(dstpath.joinpath('tresholdimg.png')), tresholdimg)
cdbscan = models.ColorDBScan([(1, lower, upper, 40, 4)])
img_rects, img_labels = cdbscan.predict([img])
draw = None
for rect in img_rects[0]:
    draw = cv.rectangle(img, dataclasses.astuple(rect), (0, 0, 255), 2)
cv.imwrite(str(dstpath.joinpath('cdbscan.png')), draw)
cv.imwrite(str(dstpath.joinpath('resizedimg.png')), resized)


#Labeled img
rects = rectangle.rects_from_json(Path('C:\\Users\\Alex\\Desktop\\UNI\\corn\\DSC_0922.json'))
labeledimg = cv.resize(fullimg.copy(), (0, 0), None, 0.5, 0.5)
for rect in rects:
    rect = rect.rescale(0.5)
    labeledimg = cv.rectangle(labeledimg, dataclasses.astuple(rect), (0, 255, 0), 3)
cv.imwrite(str(dstpath.joinpath('labeledimg.png')), labeledimg)