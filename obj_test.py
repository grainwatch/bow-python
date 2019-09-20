import cv2 as cv
import model

img = cv.imread('C:/Users/Alex\Desktop/Bilder_Korner_original_20180411/mais/DSC_0922.JPG')
hogmodel = model.HOG.load('C:/Users/Alex/CodeProjects/bow-python/hog.yml', 'C:/Users/Alex/CodeProjects/bow-python/svm.yml')
detector = model.ObjectDetector(hogmodel, (128, 128), (64, 64))
result = detector.detect(img)
pass