import argparse
from pathlib import Path

from dataset import objectdataset
import models
import viz

def main(datasetpath: Path):
    (train_imgs, train_rects, train_labels), (test_imgs, test_rects, test_labels) = objectdataset.load_grain_dataset(datasetpath)
    print('corn ap')
    #eval_color_db_scan(test_imgs, test_rects, test_labels)
    eval_hog(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels)
    #eval_slide_bow(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels)
    #eval_slide_ft_matcher(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels)

    
def eval_slide_bow(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels):
    slide_win = models.BOWSlidingWindowDetector()
    slide_win.fit(train_imgs, train_rects, train_labels)
    m_ap, aps, confusion_matrix = slide_win.evaluate(test_imgs, test_rects, test_labels)
    print(f'bow sliding window: \n\taverage precision: {m_ap}\n\tconfusion_matrix: {confusion_matrix}')
    for i in range(len(aps)):
        print(f'ap {i+1}: {aps[i]}')
    img_rects, img_labels = slide_win.predict(test_imgs)
    for img, truth_rects, truth_labels, predicted_rects, predicted_labels in zip(test_imgs, test_rects, test_labels, img_rects, img_labels):
        viz.viz_detected_objects(img, truth_rects, truth_labels, predicted_rects, predicted_labels)

def eval_slide_ft_matcher(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels):
    slide_win = models.FTMatcherSlidingWindowDetector()
    slide_win.fit(train_imgs, train_rects, train_labels)
    m_ap, aps, confusion_matrix = slide_win.evaluate(test_imgs, test_rects, test_labels)
    print(f'ft matcher sliding window: \n\taverage precision: {m_ap}\n\tconfusion_matrix: {confusion_matrix}')
    for i in range(len(aps)):
        print(f'ap {i+1}: {aps[i]}')
    img_rects, img_labels = slide_win.predict(test_imgs)
    for img, truth_rects, truth_labels, predicted_rects, predicted_labels in zip(test_imgs, test_rects, test_labels, img_rects, img_labels):
        viz.viz_detected_objects(img, truth_rects, truth_labels, predicted_rects, predicted_labels)

def eval_hog(train_imgs, train_rects, train_labels, test_imgs, test_rects, test_labels):
    slide_win = models.HOGSlidingWindowDetector(low=0.01, high=0.05)
    slide_win.fit(train_imgs, train_rects, train_labels)
    m_ap, aps, confusion_matrix = slide_win.evaluate(test_imgs, test_rects, test_labels)
    print(f'hog sliding window: \n\taverage precision: {m_ap}\n\tconfusion_matrix: {confusion_matrix}')
    for i in range(len(aps)):
        print(f'ap {i+1}: {aps[i]}')
    img_rects, img_labels = slide_win.predict(test_imgs)
    for img, truth_rects, truth_labels, predicted_rects, predicted_labels in zip(test_imgs, test_rects, test_labels, img_rects, img_labels):
        viz.viz_detected_objects(img, truth_rects, truth_labels, predicted_rects, predicted_labels)

def eval_color_db_scan(test_imgs, test_rects, test_labels):
    color_dbscan = models.ColorDBScan([(1, (15, 150, 150), (50, 255, 255), 40, 10)])
    m_ap, aps, confusion_matrix = color_dbscan.evaluate(test_imgs, test_rects, test_labels)
    print(f'color db scan: \n\taverage precision: {m_ap}\n\tconfusion_matrix: {confusion_matrix}')
    for i in range(len(aps)):
        print(f'ap {i+1}: {aps[i]}')
    img_rects, img_labels = color_dbscan.predict(test_imgs)
    for img, truth_rects, predicted_rects in zip(test_imgs, test_rects, img_rects):
        viz.viz_detected_objects(img, truth_rects, predicted_rects)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasetpath')
    args = parser.parse_args()
    main(Path(args.datasetpath))