import dataset
import model
import cv2 as cv

def _train_ft_matcher(train_x, train_y):
    ftmodel = model.FeatureMatcher()
    ftmodel.fit(train_x, train_y)
    return ftmodel

def _train_bow(train_x, train_y):
    bowmodel = model.BOWModel()
    bowmodel.fit(train_x, train_y)
    return bowmodel

def _train_hog(train_x, train_y):
    hogmodel = model.HOG()
    hogmodel.fit(train_x, train_y)
    return hogmodel

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = dataset.load_data_2('C:/Users/Alex/CodeProjects/bow-python/dataset_v2')
    #(train_x, train_y), (test_x, test_y) = dataset.filter_data(train, test, [-1, 1])
    train_x, train_y = dataset.shuffle(train_x, train_y)
    for x in train_x[:10]:
        cv.imshow('', x)
        cv.waitKey(0)
    ftmodel = _train_ft_matcher(train_x, train_y)
    bowmodel = _train_bow(train_x, train_y)
    hogmodel = _train_hog(train_x, train_y)
    ftaccuracy = ftmodel.evaluate(test_x, test_y)
    bowaccuracy = bowmodel.evaluate(test_x, test_y)
    hogaccuracy = hogmodel.evaluate(test_x, test_y)
    print(f'feature matcher accuracy: {ftaccuracy}')
    print(f'bow accuracy: {bowaccuracy}')
    print(f'hog accuracy: {hogaccuracy}')
    bowmodel.save('bow', 'bowsvm')
    hogmodel.save('hog', 'hogsvm')
