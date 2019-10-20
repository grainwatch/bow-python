from pathlib import Path
import imgclassifier
import objdetector
import datasetimg
from dataset import objectdataset
import viz

def train_hog():
    model = imgclassifier.HOGModel()
    model.fit(train_imgs, train_labels)
    score = model.evaluate(test_imgs, test_labels)
    print(score)
    save_hog_path = save_path.joinpath('hog_all_params')
    save_hog_svm_path = save_path.joinpath('hog_all_svm')
    model.save(str(save_hog_path), str(save_hog_svm_path))
    #hogdetector = objdetector.SlidingWindowDetector(model)
    #img_rects, img_labels = hogdetector.predict([test_obj_imgs[0]])
    #viz.viz_detected_objects(test_obj_imgs[0], test_obj_rects[0], test_obj_labels[0], img_rects[0], img_labels[0])
    """m_ap, aps, confusion_matrix = hogdetector.evaluate(test_obj_imgs, test_obj_rects, test_obj_labels)
    print(m_ap, confusion_matrix)"""

def train_bow():
    model = imgclassifier.BOWModel()
    model.fit(train_imgs, train_labels)
    score = model.evaluate(test_imgs, test_labels)
    print(score)
    save_bow_path = save_path.joinpath('bow_params')
    save_bow_svm_path = save_path.joinpath('bow_svm')
    model.save(str(save_bow_path), str(save_bow_svm_path))

if __name__ == '__main__':
    save_path = Path('android_models')
    save_path.mkdir(exist_ok=True)
    (train_imgs, train_labels), (test_imgs, test_labels) = datasetimg.load_data_2('C:\\Users\\Alex\\Desktop\\androiddataset128')
    #(train_obj_imgs, train_obj_rects, train_obj_labels), (test_obj_imgs, test_obj_rects, test_obj_labels) = objectdataset.load_grain_dataset('C:\\Users\\Alex\\Desktop\\fullnewdataset')
    train_hog()
    #train_bow()