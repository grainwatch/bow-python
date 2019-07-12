from pathlib import Path
import bow

import cv2 as cv

img_folder = 'C:\\Users\\Alex\\IdeaProjects\\grain-swpt\\dataset\\corn\\'

def main():
    ft_extractor = cv.ORB_create()
    img_paths = list(map(lambda x: str(x), Path(img_folder).glob('*.JPG')))[:4]
    trainer = FeatureMatcherTrainer(ft_extractor, img_paths)
    trainer.train()
    trainer.save('matcher')


class FeatureMatcherTrainer:
    def __init__(self, ft_extractor, img_paths):
        self.ft_extractor = ft_extractor
        self.img_paths = img_paths
        self.matcher = cv.FlannBasedMatcher_create()

    def train(self):
        for x in self.img_paths:
            img = cv.imread(x)
            rois = bow.get_rect_points(x)
            fts = self.ft_extractor.detectAndCompute(img, None)
            for y in rois:
                roi_fts = bow.get_features_in_roi(fts, y, 1)[1]
                print(roi_fts)
                self.matcher.add([roi_fts])

    def save(self, filename):
        print('saving')
        print(self.matcher.getTrainDescriptors())
        self.matcher.write(filename)


if __name__ == '__main__':
    main()